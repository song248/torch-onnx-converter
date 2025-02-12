from ...base import T2VRetConfig, T2VRetModelBase, T2VRetONNXConfig
from .CLIP_modules import _expand_mask
from .CLIP_VIP import CLIPVIP as clip_vip

from typing import List

import cv2
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    CenterCrop,
    Normalize,
    ToPILImage,
    ToTensor,
)
from transformers import CLIPTokenizerFast

from pia.utils.t2v_process_video import video_preprocess
from pia.utils.load_utils import load_state_dict_with_mismatch
from pia.utils.exception_utils import raise_exception_decorator


class CLIPVIP(T2VRetModelBase):
    def __init__(self, config: T2VRetConfig):
        self.config = config
        self.model = self.__load_model__()

        normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = Compose(
            [
                ToPILImage(),
                ToTensor(),
                # CenterCrop(config.img_size),
                normalize,
            ]
        )

        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_config)

    @raise_exception_decorator(FileNotFoundError)
    def _load_model(self):
        state_dict = torch.load(self.config.model_path, map_location=self.config.device)
        model = clip_vip(args=self.config, state_dict=state_dict)
        load_state_dict_with_mismatch(model, state_dict)
        if self.config.use_half_precision:
            model.eval().half()
        else:
            model.eval().float()
        model.to(device=self.config.device)
        return model


    def video_preprocess(self, video: ndarray | Tensor):
        video_tensor, video_mask_tensor = video_preprocess(
            video = video,
            device = self.config.device,
            tile_size = self.config.tile_config,
            img_size = self.config.img_size,
            transform = self.transform
        )
        
        return video_tensor


    def text_preprocess(self, texts: str | List[str]):
        if type(texts) == str:
            texts = [texts]
        ret = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_words,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = ret.input_ids.to(self.config.device)  # (B, L)
        text_input_mask = ret.attention_mask.to(self.config.device)  # (B, L)
        return text_input_ids, text_input_mask

    def preprocess(self, video: Tensor | ndarray, text: str | List[str]):
        video = self.video_preprocess(video=video)
        txt_ids, txt_mask = self.text_preprocess(texts=text)
        return video, txt_ids, txt_mask

    def encode_text(
        self,
        text_input_ids: Tensor | ndarray | str,
        text_input_mask=None,
    ):
        textual_vector = self.model.encode_text(
            text_input_ids=text_input_ids, text_input_mask=text_input_mask
        )
        return textual_vector

    def encode_video(self, video: Tensor | ndarray) -> Tensor:
        if type(video) == ndarray:
            video = self.video_preprocess(video=video)

        visual_vector = self.model.encode_video(video=video)
        return visual_vector

    @torch.no_grad
    def forward(
        self,
        video: Tensor | ndarray = None,
        text: str | List[str] | Tensor = None,
        txt_mask: Tensor = None,
    ) -> Tensor:
        if (
            type(text) == str
            or type(text) == list
            or (txt_mask is None and text is not None)
        ):
            text, txt_mask = self.text_preprocess(texts=text)

        if type(video) == ndarray:
            video = self.video_preprocess(video=video)

        if video is None:  # only text
            return self.encode_text(text_input_ids=text, text_input_mask=txt_mask)
        elif text is None:  # only video
            return self.encode_video(video=video)

        ret = self.model.forward(
            video=video, text_input_ids=text, text_input_mask=txt_mask
        )
        similarity = ret["text_features"] @ ret["vis_features"].T
        return similarity

    def export(self, onnx_config: T2VRetONNXConfig):
        dtypes = torch.float16 if onnx_config.half else torch.float32
        if onnx_config.split_part == "visual":
            new_model = VisualModel(self.model)
            input_size = {
                "input0": [
                    1,
                    self.config.temporal_size,
                    3,
                    self.config.img_size[0],
                    self.config.img_size[1],
                ]
            }
            input_dummy = torch.rand(input_size["input0"], dtype=dtypes).to(
                self.config.device
            )

        elif onnx_config.split_part == "textual":
            new_model = TextualModel(self.model)
            input_size = {
                "input0": [
                    1,
                    self.config.max_words,
                    self.model.clipmodel.text_embed_dim,
                ],
                "input1": [1, self.config.max_words],
            }
            input_dummy = (
                torch.rand(input_size["input0"], dtype=dtypes).to(self.config.device),
                torch.rand(input_size["input1"], dtype=dtypes).to(self.config.device),
            )

        elif onnx_config.split_part == "embedding":
            new_model = self.model.clipmodel.text_model.embeddings
            input_size = {
                "input0": [1, self.config.max_words],
            }
            input_dummy = torch.randint(
                0,
                self.model.clipmodel.text_model.embeddings.token_embedding.num_embeddings
                - 1,
                input_size["input0"],
                dtype=torch.int32,
            ).to(
                self.config.device
            )  # embedding = only int

        onnx_config.input_size = input_size
        onnx_config.input_dummy = input_dummy
        self.export2onnx(onnx_config, new_model)
        pass


class VisualModel(torch.nn.Module):
    def __init__(self, model: clip_vip) -> None:
        super().__init__()
        self.hidden_model = model.clipmodel.vision_model
        self.hidden_projection = model.clipmodel.visual_projection

    def forward(self, x):
        x = self.hidden_model(x)
        x = self.hidden_projection(x[1])
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class TextualModel(torch.nn.Module):
    def __init__(self, model: clip_vip) -> None:
        super().__init__()
        self.hidden_model = model.clipmodel.text_model.encoder
        self.final_layer_norm = model.clipmodel.text_model.final_layer_norm
        self.projection_layer = model.clipmodel.text_projection
        self._build_causal_attention_mask = (
            model.clipmodel.text_model._build_causal_attention_mask
        )

    def forward(self, x, mask):
        y_ex = _expand_mask(mask, x.dtype)
        if_fp16 = x.dtype == torch.float16
        bsz, seq_len = mask.size()  # (B, 50)
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, fp16=if_fp16
        ).to(x.device)
        x = self.hidden_model(
            inputs_embeds=x,
            attention_mask=y_ex,
            causal_attention_mask=causal_attention_mask,
        )
        x = self.final_layer_norm(x[0])
        x = x[torch.arange(x.shape[0]), mask.argmin(dim=-1) - 1]
        x = self.projection_layer(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
