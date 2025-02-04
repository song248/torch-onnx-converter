# torch-onnx-converter
torch export to onnx  

In this time, use Yolo v8 pt file convert to onnx file

## Model load

### Download pt file first use below link  
- https://docs.ultralytics.com/models/yolov8/  
- https://docs.ultralytics.com/quickstart/

### Environment setting
```
conda create -n onnx python=3.11 -y
conda activate onnx
pip install ultralytics
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

## Ref
https://github.com/onnx/onnx


## Clip4Clip ONNX 변환
C4C 변환시 pia-ai-package 이용 필요 (T2VRet 사용)  
그냥 clip4clip 모델 로드해서 사용해도 무방