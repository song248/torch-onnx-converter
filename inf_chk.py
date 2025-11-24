import argparse
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="비디오를 대상으로 PyTorch(.pt)와 ONNX 모델 추론 결과를 비교 저장합니다."
    )
    parser.add_argument(
        "--video",
        default="turnnel_inside_2-1_origin.mp4",
        help="추론할 영상 경로",
    )
    parser.add_argument(
        "--pt",
        default="assets/vehicle_detection_model_fine_tuned 1.pt",
        help="PyTorch 가중치 경로",
    )
    parser.add_argument(
        "--onnx",
        default="assets/vehicle_detection_model_fine_tuned 1.onnx",
        help="ONNX 모델 경로",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="추론 입력 크기",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="추론 디바이스 (예: 'cpu', '0')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="confidence 임계값",
    )
    parser.add_argument(
        "--project",
        default="runs/inf_chk",
        help="YOLO가 결과를 저장할 기본 폴더",
    )
    return parser.parse_args()


def check_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} 파일을 찾을 수 없습니다: {path}")


def run_inference(
    model_path: Path,
    source_path: Path,
    project: Path,
    name: str,
    imgsz: int,
    device: str,
    conf: float,
) -> Tuple[Path, Path]:
    print(f"[{name}] 모델 로드: {model_path}")
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=True,
        save_conf=True,
        project=str(project),
        name=name,
        exist_ok=True,
        vid_stride=1,
    )
    save_dir = Path(results[0].save_dir) if results else project / name
    video_out = save_dir / source_path.name
    print(f"[{name}] 추론 완료. 저장 경로: {video_out}")
    return save_dir, video_out


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    pt_path = Path(args.pt)
    onnx_path = Path(args.onnx)
    project_path = Path(args.project)

    check_exists(video_path, "비디오")
    check_exists(pt_path, "PyTorch 가중치")
    check_exists(onnx_path, "ONNX 모델")

    print("PyTorch 모델 추론 시작")
    _, pt_video = run_inference(
        pt_path, video_path, project_path, "pt", args.imgsz, args.device, args.conf
    )

    print("ONNX 모델 추론 시작")
    _, onnx_video = run_inference(
        onnx_path, video_path, project_path, "onnx", args.imgsz, args.device, args.conf
    )

    print("=== 추론 결과 ===")
    print(f"PyTorch 결과: {pt_video}")
    print(f"ONNX 결과:    {onnx_video}")


if __name__ == "__main__":
    main()
