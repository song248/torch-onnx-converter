import onnx
import onnxruntime as ort
import numpy as np

# ONNX model load
onnx_file_path = './assets/yolov8m.onnx'
onnx_model = onnx.load(onnx_file_path)

# validate ONNX model
onnx.checker.check_model(onnx_model)
print("Success convert to ONNX")

# ONNX runtime session initialize
ort_session = ort.InferenceSession(onnx_file_path)

# ONNX runtime inference
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = ort_session.run(None, {"images": input_data})
print("outputs:", outputs)