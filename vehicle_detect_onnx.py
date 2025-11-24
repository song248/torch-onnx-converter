import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned vehicle detection YOLO checkpoint to ONNX."
    )
    parser.add_argument(
        "--weights",
        default="assets/vehicle_detection_model_fine_tuned 1.pt",
        help="경로를 지정하지 않으면 기본 차량 탐지 가중치를 사용합니다.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="저장할 ONNX 파일 경로. 기본값은 weights 파일명에 .onnx 확장자를 붙인 경로입니다.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="모델이 학습된 입력 크기. 기본값 960.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="모델을 로드하고 내보낼 디바이스(e.g. 'cpu', '0').",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset 버전.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="지정하면 dynamic axes 없이 고정 입력 크기로 내보냅니다.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="지정하면 ONNX simplify 단계를 생략합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"가중치 파일을 찾을 수 없습니다: {weights_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = weights_path.with_suffix(".onnx")

    print(f"YOLO 가중치 로드 중: {weights_path}")
    model = YOLO(str(weights_path))
    model_meta = getattr(model.model, "names", None)
    model_nc = getattr(model.model, "nc", None)
    if model_meta is not None and model_nc is not None:
        print(f"모델 클래스 수: {model_nc} / 클래스 이름: {model_meta}")

    export_kwargs = {
        "format": "onnx",
        "opset": args.opset,
        "dynamic": not args.static,
        "simplify": not args.no_simplify,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    print(
        f"ONNX 변환 시작 -> 입력 크기 {args.imgsz}, "
        f"dynamic={export_kwargs['dynamic']}, simplify={export_kwargs['simplify']}"
    )
    exported_path = Path(model.export(**export_kwargs))

    if output_path.resolve() != exported_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(output_path)
        exported_path = output_path

    print(f"ONNX 변환 완료: {exported_path}")


if __name__ == "__main__":
    main()
