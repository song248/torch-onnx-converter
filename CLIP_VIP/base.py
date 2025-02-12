import os
from abc import abstractmethod
from typing import List, Union, final, Literal

import numpy as np
import torch
from pia.base import PiaConfigBase, PiaModelBase, PiaONNXConfigBase
from pia.utils.exception_utils import validate_tile_config


class T2VRetConfig(PiaConfigBase):
    """
    Initialize T2VRetConfig with text-to-video retrieval model settings.

    Args:
        model_path (str): Model path
        device (str, optional): Device (default: "cpu")
        use_half_precision (bool, optional): Half precision (default: False)
        img_size (List[int], optional): Image size (default: [224, 224])
        max_words (int, optional): Maximum words (default: 32)
        temporal_size (int, optional): Temporal size (default: 12)
        frame_skip (int, optional): Frame skip (default: 15)
        clip_config (str, optional): CLIP config (default: "openai/clip-vit-base-patch32")

    Examples:
        Initialize T2VRetConfig with default settings:

        >>> config = T2VRetConfig(model_path="model.pth")
        >>> model = PiaTorchModel(target_task="RET", target_model="clip4clip", config=config)
        >>>
        >>> video = np.random.rand(3, 12, 224, 224, 3)
        >>> text = "a photo of a cat"
        >>>
        >>> similiarity = model.forward(video, text)

        Initialize T2VRetConfig with custom settings:

        >>> config = T2VRetConfig(
        ...     model_path="model.pth",
        ...     device="cuda",
        ...     use_half_precision=True,
        ...     img_size=[256, 256],
        ...     max_words=64,
        ...     temporal_size=24,
        ...     frame_skip=30,
        ...     clip_config="openai/clip-vit-base-patch32",
        ... )
        >>> model = PiaTorchModel(target_task="RET", target_model="clip4clip", config=config)
        >>>
        >>> video = np.random.rand(3, 24, 256, 256, 3)
        >>> text = "a photo of a cat"
        >>>
        >>> similiarity = model.forward(video, text)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        tile_config: Literal["L", "M", "S"] = None,
        img_size: List[int] = [224, 224],
        max_words: int = 32,
        temporal_size: int = 12,
        frame_skip: int = 15,
        clip_config="openai/clip-vit-base-patch32",
    ):
        # if tile_config:
        #     validate_tile_config(tile_config)

        super().__init__(model_path, device, use_half_precision, tile_config)
        self.img_size = img_size
        self.max_words = max_words
        self.temporal_size = temporal_size
        self.frame_skip = frame_skip

        # clip-vip config
        self.clip_config = clip_config

        # clip4clip config
        self.loose_type = True
        self.linear_patch = "2d"
        self.sim_header = "meanP"
        self.initializer_range = 0.02
        self.max_position_embeddings = 128


class T2VRetONNXConfig(PiaONNXConfigBase):
    def __init__(
        self,
        output_dir: str,
        split_part: str,  # visual, textual, embedding
        output_cnt: int = 1,
        opset: int = 14,  # if error occured - change opset version 8 ~ 20
        half: bool = False,
    ):
        """
        : params :
            split_part : [visual, textual, embedding]
        """
        super().__init__(output_dir, opset, half)
        self.split_part = split_part
        self.output_cnt = output_cnt
        self.input_size = {}
        self.input_dummy = {}


class T2VRetModelBase(PiaModelBase):
    @abstractmethod
    def _load_model(self, model_path: str):
        pass

    @abstractmethod
    def preprocess(
        self,
        video: Union[torch.Tensor, np.ndarray],
        text: Union[str, List[str]],
    ):
        pass

    @abstractmethod
    # @torch.no_grad - overriding
    def forward(
        self,
        video: Union[torch.Tensor, np.ndarray],
        text: Union[torch.Tensor, np.ndarray, str, List[str]],
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    @final
    def export2onnx(self, onnx_config: T2VRetONNXConfig, model):
        from pia.model import PiaONNXTensorRTModel  # partial import error

        input_names = list(onnx_config.input_size.keys())
        output_names = ["output" + str(i) for i in range(onnx_config.output_cnt)]
        dynamic_axes = {}
        for name in input_names + output_names:
            dynamic_axes[name] = {0: "batch_size"}

        input_dummy_data = onnx_config.input_dummy

        export_params = dict(
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            verbose=False,
            opset_version=onnx_config.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
        model_file_name = os.path.splitext(os.path.basename(self.config.model_path))[0]
        save_path = os.path.join(
            onnx_config.output_dir,
            f"{model_file_name}_{onnx_config.split_part}_{'f16' if onnx_config.half else 'f32'}_op{onnx_config.opset}.onnx",
        )
        torch.onnx.export(model, input_dummy_data, save_path, **export_params)

        export_model = PiaONNXTensorRTModel(
            model_path=save_path,
            device=self.config.device,
            half=onnx_config.half,
        )

        ret = export_model(input_dummy_data)
        if len(input_names) != 1:
            origin_output = model(*input_dummy_data)
        else:
            origin_output = model(input_dummy_data)

        torch.testing.assert_close(origin_output, ret, atol=1e-3, rtol=1e-3)
        print(f"diff sum : {torch.sum(torch.abs(origin_output - ret)):.6f}")
