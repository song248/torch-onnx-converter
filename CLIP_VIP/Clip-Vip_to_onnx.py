import torch
import torch.onnx
from CLIP_VIP.CLIP_VIP import CLIPVIP  # CLIPVIP가 정의된 파일을 import
from CLIP_VIP.base import T2VRetConfig  # 모델 설정 클래스
import onnx
import platform


def get_device(device: str = None) -> str:
    """단일 디바이스를 문자열로 리턴한다. None이 입력으로 들어올 시 자동으로 적합한 device를 지정한다."""
    if device == "cpu":
        return "cpu"

    if device is None:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    if device == "gpu" and torch.cuda.is_available():
        return "cuda"

    if "cuda" in device:
        if torch.cuda.is_available():
            return device
        else:
            return "cpu"

    if device == "mps":
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    raise ValueError(f"유효하지 않은 디바이스: {device}. 사용 가능한 디바이스는 {VALID_DEVICE}입니다.")


# 모델 설정 및 더미 입력 생성
def load_model_and_convert_to_onnx(state_dict_path, output_onnx_path):
    # 사전 학습된 가중치 로드
    state_dict = torch.load(state_dict_path, map_location="cpu")

    # 모델 설정 값 (예제 설정, 실제 환경에 맞게 변경 필요)
    args = T2VRetConfig(
        model_path=str("./assets/pretrain_clipvip_base_32.pt"),
        device=get_device(),
        use_half_precision=False,
    )

    # 모델 로드
    model = CLIPVIP(args, state_dict)
    model.eval()  # 평가 모드 설정

    # 더미 입력 생성 (ONNX 변환을 위해 필요)
    batch_size = 1
    num_frames = 8  # 예제 값
    channels, height, width = 3, 224, 224
    seq_length = 77  # 텍스트 입력 길이 (CLIP 기본값)ㅋ

    dummy_video = torch.randn(batch_size, num_frames, channels, height, width)
    dummy_text_ids = torch.randint(0, 49408, (batch_size, seq_length))  # CLIP vocab size 기준
    dummy_text_mask = torch.ones(batch_size, seq_length)

    # ONNX 변환
    torch.onnx.export(
        model, 
        (dummy_video, dummy_text_ids, dummy_text_mask),  # 입력값 튜플
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["video", "text_input_ids", "text_input_mask"],
        output_names=["text_features", "vis_features"],
        dynamic_axes={
            "video": {0: "batch_size", 1: "num_frames"},
            "text_input_ids": {0: "batch_size", 1: "seq_length"},
            "text_input_mask": {0: "batch_size", 1: "seq_length"},
            "text_features": {0: "batch_size"},
            "vis_features": {0: "batch_size"},
        }
    )

    print(f"ONNX 모델 변환 완료: {output_onnx_path}")

# 실행 예시
state_dict_path = "./assets/pretrain_clipvip_base_32.pt"
output_onnx_path = "clipvip_model.onnx"

load_model_and_convert_to_onnx(state_dict_path, output_onnx_path)
