# pe_pt_to_onnx.py  (vlm_t 구조 대응, robust visual-only export)

import os
import sys
import argparse
import torch
import torch.nn as nn

# ---- 패키지 경로 보정 (로컬 실행 편의) ----
CUR = os.path.dirname(os.path.abspath(__file__))
if CUR not in sys.path:
    sys.path.insert(0, CUR)

# vlm_t/vision_encoder 모듈 임포트
from vlm_t.vision_encoder import config as pe_cfg      # PE_VISION_CONFIG / PE_TEXT_CONFIG  :contentReference[oaicite:5]{index=5}
from vlm_t.vision_encoder.pe import CLIP              # CLIP / VisionTransformer / 로더들  :contentReference[oaicite:6]{index=6}

def parse_args():
    p = argparse.ArgumentParser(description="Convert PE ckpt to ONNX (visual-only by default)")
    p.add_argument("--ckpt", default="assets/FT_PE-Core-L14-336_250804.ckpt",
                   help="Path to fine-tuned checkpoint (.ckpt/.pt/.pth)")
    p.add_argument("--onnx", default="assets/model/PE-Core-L14-336_vision_dynamic.onnx",
                   help="Output ONNX path")
    p.add_argument("--model_name", default="PE-Core-L14-336",
                   help="Base model name (see vlm_t/vision_encoder/config.py)")
    p.add_argument("--split_qkv", action="store_true",
                   help="Use split-qkv variant (appends '-splitqkv' to model name)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dynamic_hw", action="store_true",
                   help="Make H/W dynamic in addition to batch")
    return p.parse_args()

def is_vision_only_state_dict(sd_keys):
    """
    텍스트 타워 핵심 키(token_embedding, positional_embedding 등)가 거의 없고
    'visual.' 접두사 또는 비전 블록 키가 많으면 비전 전용으로 간주.
    """
    s = list(sd_keys)
    has_text_core = any(("token_embedding" in k or "positional_embedding" in k) for k in s)
    mostly_visual = any(k.startswith("visual.") for k in s) or any(
        k.startswith(("conv1.", "transformer.resblocks")) for k in s
    )
    return (not has_text_core) and mostly_visual

class VisualWrapper(nn.Module):
    """ONNX 내보내기용: (B,3,H,W) -> (B,D)"""
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.visual = clip_model.visual  # VisionTransformer  :contentReference[oaicite:7]{index=7}
    def forward(self, x):
        return self.visual(x)            # 비전 인코더 출력  :contentReference[oaicite:8]{index=8}

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)

    # 0) 모델 이름 구성 (split-qkv 변형 지원)
    cfg_name = a.model_name + ("-splitqkv" if a.split_qkv else "")
    if cfg_name not in pe_cfg.PE_VISION_CONFIG or cfg_name not in pe_cfg.PE_TEXT_CONFIG:
        raise ValueError(f"Unknown config: {cfg_name} (check vlm_t/vision_encoder/config.py)")

    # 1) 베이스 CLIP 초기화 (pretrained=False)
    model = CLIP.from_config(cfg_name, pretrained=False).to(a.device).eval()  # :contentReference[oaicite:9]{index=9}

    # 2) ckpt 로드 (두 경로 시도)
    print(f"[INFO] Loading ckpt: {a.ckpt}")
    sd = torch.load(a.ckpt, map_location=a.device)
    if isinstance(sd, dict) and ("state_dict" in sd or "weights" in sd or "model" in sd):
        # 보편적 래핑 해제
        sd = sd.get("state_dict", sd.get("weights", sd.get("model")))
    if isinstance(sd, dict):
        # DDP 접두사 제거
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # (A) CLIP 전체에 strict=False로 시도
    missing = unexpected = []
    try:
        m, u = model.load_state_dict(sd, strict=False)  # 유연 로드  :contentReference[oaicite:10]{index=10}
        missing, unexpected = list(m), list(u)
        print(f"[INFO] Loaded to CLIP (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[WARN] Loading to CLIP failed: {e}")

    # (B) 비전 전용이면 비전 타워에만 로드 (접두사 정리 + strict=False)
    if is_vision_only_state_dict(sd.keys()) or len(missing) > 1000:
        print("[INFO] Detected vision-only checkpoint. Loading into vision tower...")
        vis_sd = {}
        for k, v in sd.items():
            if k.startswith("visual."):
                vis_sd[k.replace("visual.", "", 1)] = v
            else:
                vis_sd[k] = v
        # pe.py의 비전 로더는 strict=False를 권장 (키 불일치 허용)  :contentReference[oaicite:11]{index=11}
        m, u = model.visual.load_state_dict(vis_sd, strict=False)
        print(f"[INFO] Visual-only load done. missing={len(m)} unexpected={len(u)}")

    # 3) 더미 입력 (image_size는 CLIP이 들고 있음)
    H = W = int(model.image_size)   # CLIP이 내부에서 visual.image_size를 노출  :contentReference[oaicite:12]{index=12}
    dtype = torch.float16 if a.fp16 else torch.float32
    dummy = torch.randn(1, 3, H, W, device=a.device, dtype=dtype)

    # 4) ONNX Export (비전 타워만 래핑)
    wrapper = VisualWrapper(model).to(a.device).eval()
    dynamic_axes = {"input": {0: "batch"}}
    if a.dynamic_hw:
        dynamic_axes["input"].update({2: "height", 3: "width"})

    print(f"[INFO] Exporting ONNX -> {a.onnx}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            a.onnx,
            export_params=True,
            opset_version=a.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    sz = os.path.getsize(a.onnx) / (1024 * 1024)
    print(f"[OK] ONNX saved: {a.onnx}  ({sz:.2f} MB)")
    print(f"[INFO] image_size={H}  fp16={a.fp16}  dynamic_hw={a.dynamic_hw}")

if __name__ == "__main__":
    main()
