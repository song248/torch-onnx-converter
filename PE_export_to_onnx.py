"""
python export_to_onnx.py \                  
  --output model/FT_PE-Core-L14-336_250804_merged.onnx \
  --split-qkv \
  --pretrained-split-qkv-path model/PE-Core-L14-336-splitqkv.pt
"""
import argparse
import json
import os
from typing import Dict, Iterable

import torch

from PE.PE_class import PEModelInitializer
from peft import LoraConfig, PeftModel, get_peft_model


KNOWN_PREFIXES_TO_STRIP: tuple[str, ...] = ("module.",)


def _extract_state_dict(state_obj: Dict) -> Dict:
    if isinstance(state_obj, dict):
        for key in ("state_dict", "model", "weights", "module"):
            maybe = state_obj.get(key)
            if isinstance(maybe, dict):
                return maybe
    return state_obj


def _strip_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _resolve_adapter_config_path(weight_path: str | None, adapter_dir: str | None) -> str | None:
    candidates = []
    if adapter_dir:
        candidates.append(os.path.join(adapter_dir, "adapter_config.json"))
    if weight_path:
        base_dir = os.path.dirname(weight_path)
        candidates.append(os.path.join(base_dir, "adapter_config.json"))
        filename = os.path.basename(weight_path)
        parts = filename.split(".")
        if parts and parts[-1].isdigit():
            candidates.append(os.path.join(base_dir, f"lora_adapter.bin.{parts[-1]}", "adapter_config.json"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_lora_config(config_path: str) -> LoraConfig:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias=cfg.get("bias", "none"),
        modules_to_save=cfg.get("modules_to_save"),
        task_type=cfg.get("task_type", "FEATURE_EXTRACTION"),
    )


def _load_lora_weight_with_cleanup(initializer: PEModelInitializer) -> torch.nn.Module:
    initializer.load_base_model(pretrained=True)

    adapter_config_path = _resolve_adapter_config_path(initializer.weight_path, initializer.lora_adapter_path)
    if not adapter_config_path:
        raise FileNotFoundError(
            "Could not locate adapter_config.json. Provide --adapter-dir pointing to the LoRA adapter folder."
        )

    lora_config = _load_lora_config(adapter_config_path)

    # LoRA가 q_proj/k_proj/v_proj를 대상으로 한다면 split_qkv 모델이 필요하다.
    qkv_targets = {"q_proj", "k_proj", "v_proj"}
    if qkv_targets.intersection(set(lora_config.target_modules or [])) and not initializer.split_qkv:
        print("[INFO] Detected q/k/v projector LoRA targets; reloading base model with split_qkv=True.")
        initializer.split_qkv = True
        if not initializer.pretrained_split_qkv_path or not os.path.isfile(initializer.pretrained_split_qkv_path):
            raise FileNotFoundError(
                "split_qkv base weights required. Supply --pretrained-split-qkv-path pointing to "
                "the PE-Core-L14-336-splitqkv checkpoint."
            )
        initializer.load_base_model(pretrained=True)

    initializer.model = get_peft_model(initializer.model, lora_config)

    if not initializer.weight_path or not os.path.isfile(initializer.weight_path):
        raise FileNotFoundError(f"LoRA weight file not found: {initializer.weight_path}")

    state: Dict = torch.load(initializer.weight_path, map_location=initializer.device)
    state = _extract_state_dict(state)
    state = _strip_prefixes(state, KNOWN_PREFIXES_TO_STRIP)

    incompatible = initializer.model.load_state_dict(state, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing:
        preview = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        print(f"[WARN] Missing keys during checkpoint load: {preview}")
    if unexpected:
        preview = ", ".join(unexpected[:6]) + (" ..." if len(unexpected) > 6 else "")
        print(f"[WARN] Unexpected keys during checkpoint load: {preview}")

    return initializer.model


class VideoEncoderWrapper(torch.nn.Module):
    """
    Thin wrapper so the exported ONNX graph matches the runtime call in main.py:
    frames → video embedding vector.
    """

    def __init__(self, base_model: torch.nn.Module, normalize: bool = False):
        super().__init__()
        self.base_model = base_model
        self.normalize = normalize

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: float32 tensor shaped (B, T, 3, H, W) that is already resized to
               the model's image_size and normalized with mean/std = 0.5.
        """
        return self.base_model.encode_video(video, normalize=self.normalize)


def maybe_merge_lora(model):
    """
    Loads the LoRA adapter weights, merges them into the base model, and removes
    the PEFT wrappers so that the merged weights can be cleanly exported.
    """
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def export_video_encoder(
    model_name: str,
    load_type: str,
    checkpoint: str | None,
    adapter_dir: str | None,
    split_qkv: bool,
    pretrained_split_qkv_path: str | None,
    output_path: str,
    temporal_size: int,
    image_size: int,
    normalize_output: bool,
    opset: int,
):
    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    if adapter_dir and not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
    if split_qkv and not pretrained_split_qkv_path:
        raise ValueError("split_qkv=True requires --pretrained-split-qkv-path pointing to the base checkpoint.")

    initializer = PEModelInitializer(
        model_name=model_name,
        device=str(device),
        load_type=load_type,
        weight_path=checkpoint,
        lora_adapter_path=adapter_dir,
        split_qkv=split_qkv,
        pretrained_split_qkv_path=pretrained_split_qkv_path,
    )

    if load_type == "lora_weight_load":
        clip = _load_lora_weight_with_cleanup(initializer)
    else:
        clip, *_ = initializer.initialize()

    clip = maybe_merge_lora(clip).to(device)
    clip.eval()
    wrapper = VideoEncoderWrapper(clip, normalize=normalize_output)

    dummy = torch.randn(1, temporal_size, 3, image_size, image_size, device=device)
    dynamic_axes = {
        "video": {0: "batch", 1: "frames"},
        "video_embedding": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["video"],
        output_names=["video_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[INFO] ONNX export complete → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PE video encoder to ONNX")
    parser.add_argument("--model-name", default="PE-Core-L14-336", help="Config key for CLIP backbone")
    parser.add_argument("--load-type", default="lora_weight_load", choices=[
        "default", "weight_load", "lora_weight_load", "lora_adapter_load", "lora_weight_head_load"
    ], help="Loading mode to mirror PEModelInitializer behaviour")
    parser.add_argument("--checkpoint", default="model/FT_PE-Core-L14-336_250804.pt", help="Path to fine-tuned .pt/.bin weights (if required)")
    parser.add_argument("--adapter-dir", default="FT_PE-Core-L14-336_250804_adapter", help="LoRA adapter directory (empty string to skip)")
    parser.add_argument("--split-qkv", action="store_true", help="Whether to load the split-QKV variant of the backbone")
    parser.add_argument("--pretrained-split-qkv-path", default="", help="Checkpoint with the base split-QKV weights (required if --split-qkv)")
    parser.add_argument("--output", default="model/FT_PE-Core-L14-336_250804.onnx", help="Destination ONNX file")
    parser.add_argument("--temporal-size", type=int, default=8, help="Number of frames used during export")
    parser.add_argument("--image-size", type=int, default=336, help="Input resolution fed into the vision encoder")
    parser.add_argument("--normalize-output", action="store_true", help="Apply L2 normalization to the exported features")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = args.adapter_dir or None
    export_video_encoder(
        model_name=args.model_name,
        load_type=args.load_type,
        checkpoint=args.checkpoint,
        adapter_dir=adapter_dir,
        split_qkv=args.split_qkv,
        pretrained_split_qkv_path=args.pretrained_split_qkv_path or None,
        output_path=args.output,
        temporal_size=args.temporal_size,
        image_size=args.image_size,
        normalize_output=args.normalize_output,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
