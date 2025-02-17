import torch
from ultralytics import YOLO

model = YOLO("./assets/best_ckpt.pt")
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)

onnx_file_path = "yolov8m_best.onnx"
model.export(format="onnx", opset=12, dynamic=True, simplify=True)

print(f"모델이 {onnx_file_path}로 변환되었습니다.")