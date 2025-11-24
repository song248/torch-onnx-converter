import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO PyTorch checkpoint and ONNX outputs.")
    parser.add_argument(
        "--weights",
        default="assets/vehicle_detection_model_fine_tuned 1.pt",
        help="비교할 PyTorch 가중치(.pt)",
    )
    parser.add_argument(
        "--onnx",
        default="assets/vehicle_detection_model_fine_tuned 1.onnx",
        help="검증할 ONNX 모델 경로",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="입력 이미지 크기(정사각형). 기본값 960",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="배치 크기",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch 추론 디바이스(e.g. 'cpu', '0')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="입력 난수 시드",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="np.allclose 상대 오차 허용치",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="np.allclose 절대 오차 허용치",
    )
    return parser.parse_args()


def load_pt_outputs(weights_path: Path, input_data: np.ndarray, device: str) -> np.ndarray:
    model = YOLO(str(weights_path))
    model.model.to(device).eval()
    torch_input = torch.from_numpy(input_data).to(device)
    with torch.no_grad():
        torch_outputs = model.model(torch_input)
    if isinstance(torch_outputs, (list, tuple)):
        torch_outputs = torch_outputs[0]
    return torch_outputs.detach().cpu().numpy()


def load_onnx_outputs(onnx_path: Path, input_data: np.ndarray) -> np.ndarray:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    onnx_path = Path(args.onnx)

    if not weights_path.exists():
        raise FileNotFoundError(f"PyTorch 가중치를 찾을 수 없습니다: {weights_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {onnx_path}")

    rng = np.random.default_rng(args.seed)
    input_shape = (args.batch, 3, args.imgsz, args.imgsz)
    input_data = rng.standard_normal(input_shape, dtype=np.float32)

    print(f"PyTorch 추론 시작 (weights={weights_path}, device={args.device})")
    torch_out = load_pt_outputs(weights_path, input_data.copy(), args.device)
    print(f"PyTorch 출력 shape: {torch_out.shape}")

    print(f"ONNX 추론 시작 (onnx={onnx_path})")
    onnx_out = load_onnx_outputs(onnx_path, input_data.copy())
    print(f"ONNX 출력 shape: {onnx_out.shape}")

    if torch_out.shape != onnx_out.shape:
        raise ValueError(f"출력 shape 불일치: torch {torch_out.shape} vs onnx {onnx_out.shape}")

    diff = np.abs(torch_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    allclose = np.allclose(torch_out, onnx_out, rtol=args.rtol, atol=args.atol)

    print(f"최대 절대 오차: {max_diff:.6f}")
    print(f"평균 절대 오차: {mean_diff:.6f}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}) => {allclose}")


if __name__ == "__main__":
    main()
