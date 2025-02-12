import onnxruntime as ort
import numpy as np

onnx_model_path = './assets/CLIP-ViP_MSRVTT.onnx'
session = ort.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name
# (batch, frames=12, channels=3, height=224, width=224)
input_data = np.random.randn(1, 12, 3, 224, 224).astype(np.float32)
print("Input shape:", input_data.shape)

outputs = session.run([], {input_name: input_data})
print(outputs)