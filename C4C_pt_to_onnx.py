import os
import torch
import numpy as np
import onnx
import onnxruntime as ort

from models import VisualModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.model import PiaTorchModel
from utils import non_normalized_transform

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    pt_file_name = "epoch50-pytorch_model_bin_30.pt"
    onnx_file_name = "epoch50-pytorch_model_bin_30.onnx"

    pt_file_path = os.path.join(current_dir, pt_file_name)
    onnx_save_path = os.path.join(current_dir, onnx_file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ret_config = T2VRetConfig(
        model_path=pt_file_path, 
        device="cpu",
        img_size=[224, 224],
        max_words=32,
        temporal_size=12,
        frame_skip=15,
        clip_config="openai/clip-vit-base-patch32"
    )

    ret_model = PiaTorchModel(
        target_task="RET",
        target_model=0,
        config=ret_config
    )
    visual_model = VisualModel(ret_model).float().to(device)

    dummy_input_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_input_torch = non_normalized_transform(dummy_input_np)[None, :].to(device)

    with torch.no_grad():
        output_torch = visual_model(dummy_input_torch)
    # output_torch_np = output_torch.numpy()
    output_torch_np = output_torch.cpu().numpy()

    torch.onnx.export(
        visual_model,
        dummy_input_torch,
        onnx_save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=17
    )
    print(f"ONNX 모델이 저장되었습니다: {onnx_save_path}")

    # ONNX 모델 유효성 검사
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 유효성 검사를 통과했습니다.")

    # ONNX Runtime 추론
    ort_session = ort.InferenceSession(onnx_save_path)
    ort_inputs = {
        ort_session.get_inputs()[0].name: dummy_input_torch.cpu().numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output_np = ort_outs[0]

    # PyTorch vs. ONNX
    check = np.allclose(onnx_output_np, output_torch_np, atol=1e-05, rtol=1e-04)
    if check:
        print("PyTorch 추론 결과와 ONNX 추론 결과가 유사합니다. (allclose)")
    else:
        print("경고: PyTorch 결과와 ONNX 결과가 다릅니다.")

if __name__ == "__main__":
    main()
