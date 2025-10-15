import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from peft import get_peft_model, LoraConfig, PeftModel
from collections import OrderedDict

# Safe fallback (only needed if running from deep subfolder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PE.vision_text_head import VisionTextHead
import PE.vision_encoder.pe as pe
import PE.vision_encoder.transforms as transforms


class PEModelInitializer:
    def __init__(self,
                args = None,   
                model_name: str = 'PE-Core-L14-336',
                device: str = 'cuda:0',
                load_type: str = 'default',
                weight_path = None,
                lora_adapter_path = None,
                split_qkv: bool = False,
                pretrained_split_qkv_path = None):
        
        assert load_type in ['default', 'weight_load', 'lora_weight_load', 'lora_adapter_load', 'lora_weight_head_load'], f"Invalid load_type: {load_type}"
        self.args = args
        self.model_name = model_name
        self.device = device
        self.load_type = load_type
        self.weight_path = weight_path
        self.lora_adapter_path = lora_adapter_path
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.split_qkv = split_qkv
        self.pretrained_split_qkv_path = pretrained_split_qkv_path


        # Define self.constraints for each mode
        self.constraints = {
            "default": {
                "weight_path": False,
                "lora_adapter_path": False,
            },
            "weight_load": {
                "weight_path": "required",
                "lora_adapter_path": False,
            },
            "lora_weight_load": {
                "lora_adapter_path": False,
            },
            "lora_adapter_load": {
                "lora_adapter_path": "required",
                "weight_path": False,
            },
            "lora_weight_head_load": {
                "args": "required",
                "pretrained_split_qkv_path": "required"            
                },
        }


    def load_base_model(self, pretrained: bool = True):
        if self.split_qkv:
            self.model_name = self.model_name + "-splitqkv"
            print(f"Model name: {self.model_name}")
            self.model = pe.CLIP.from_config(self.model_name, pretrained=False).to(self.device)
            if pretrained:
                state_dict = torch.load(self.pretrained_split_qkv_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights with split QKV: {self.pretrained_split_qkv_path}")

        else:
            self.model = pe.CLIP.from_config(self.model_name, pretrained=pretrained).to(self.device)

    def load_fine_tuned_weights(self):
        self.load_base_model(pretrained=False)
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Fine-tuned weights not found: {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        suffix = "_".join(self.weight_path.split("/")[-2:])
        self.model_name = f"{self.model_name}_{suffix}"

    def load_lora_weight(self,):
        self.load_base_model(pretrained=True)
        if self.load_type == "lora_weight_head_load" and self.weight_path:

            lora_args = self.args.training.lora
            if lora_args.use:
                print("Use the LoRA config from args.training.lora; Ensure it same with the current model")
                lora_config = LoraConfig(
                    r=lora_args.rank,
                    lora_alpha=lora_args.alpha,
                    target_modules=lora_args.target_modules,
                    lora_dropout=lora_args.dropout,
                    bias=lora_args.bias,
                    modules_to_save=lora_args.modules_to_save,
                    task_type="FEATURE_EXTRACTION"
                )
            else:
                print("Please modify the args.training.lora as it will be use for LoRA config. Ensure it same from your fine-tuned weight")
                raise NotImplementedError
            self.model = get_peft_model(self.model, lora_config)            
        elif self.load_type == "lora_weight_load" and self.weight_path:
            # Eval mode
            folder_path, filename = os.path.split(self.weight_path)
            epoch_number = int(filename.split(".")[-1])
            adapter_config_path = os.path.join(folder_path, f"lora_adapter.bin.{epoch_number}", "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

            with open(adapter_config_path, "r") as f:
                config_json = json.load(f)

            lora_config = LoraConfig(
                r=config_json["r"],
                lora_alpha=config_json["lora_alpha"],
                target_modules=config_json["target_modules"],
                lora_dropout=config_json.get("lora_dropout", 0.0),
                bias=config_json.get("bias", "none"),
                modules_to_save=config_json.get("modules_to_save", None),
                task_type=config_json.get("task_type", "FEATURE_EXTRACTION")
            )

            self.model = get_peft_model(self.model, lora_config)

            # Load LoRA weights
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        else: 
            # Training mode
            pass


    def load_lora_adapter(self):
        self.load_base_model(pretrained=True)
        if not os.path.exists(self.pretrained_split_qkv_path):
            raise FileNotFoundError(f"{self.pretrained_split_qkv_path} does not exist.")
        
        adapter_config_path = os.path.join(self.lora_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")
        self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)


    def _load_weight_with_head(self, model_with_head, strict=False):
        """
        Load weights into `model_with_head` from `weight_path`, handling common checkpoint formats.
        Returns (missing_keys, unexpected_keys).
        """
        if not os.path.isfile(self.weight_path):
            raise FileNotFoundError(f"Weight file not found: {self.weight_path}")

        ckpt = torch.load(self.weight_path, map_location=self.device)

        # Support common wrappers: plain state_dict, {'state_dict': ...}, {'model': ...}
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                # assume it's already a state_dict
                state = ckpt
        else:
            # very rare, but keep fallback
            state = ckpt

        # Strip DistributedDataParallel prefixes if present
        cleaned = OrderedDict()
        for k, v in state.items():
            if k.startswith("module."):
                cleaned[k[len("module."):]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model_with_head.load_state_dict(cleaned, strict=strict)
        return missing, unexpected
 

    def setup_transforms(self):
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        self.max_words = self.model.context_length
        self.image_resolution = self.model.image_size

    def initialize(self):

        # Validate self.constraints if load_type exists
        if self.load_type not in self.constraints:
            raise ValueError(f"Unknown load_type: {self.load_type}")

        for attr, expected in self.constraints.get(self.load_type, {}).items():
            value = getattr(self, attr)
            if expected == "required":
                assert value, f"{attr} should not be None/False for {self.load_type} load_type"
            elif expected is None:
                assert value is None, f"{attr} should be None for {self.load_type} load_type"
            elif expected is False:
                assert not value, f"{attr} should be False/empty for {self.load_type} load_type"

        # Mode-specific loading logic
        if self.load_type == "default":
            self.load_base_model(pretrained=True)
            print("Loaded base model")
            self.setup_transforms()

        elif self.load_type == "weight_load":
            self.load_fine_tuned_weights()
            print("Loaded fine-tuned model")
            self.setup_transforms()

        elif self.load_type == "lora_weight_load":            
            self.load_lora_weight()
            print("Loaded LoRA with weights from lora_weight_load")
            self.setup_transforms()

        elif self.load_type == "lora_adapter_load":
            self.load_lora_adapter()
            print("Loaded LoRA with Adapter")
            self.setup_transforms()
        
        elif self.load_type == "lora_weight_head_load":            
            self.load_lora_weight()
            self.setup_transforms()
            print("Loaded LoRA with weights from lora_weight_head_load")
            base_model = self.model
            model = VisionTextHead(self.args, base_model, self.device)
            print("Loaded VisionTextHead layer")
            if self.weight_path:
                missing, unexpected = self._load_weight_with_head(model_with_head = model,
                                                                  strict=True)
                print("Loaded VisionTextHead weight")
            del self.model
            self.model = model
        
        return self.model, self.preprocess, self.tokenizer, self.max_words, self.image_resolution

