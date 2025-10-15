# pe_pt_to_onnx.py  — Export PE ckpt to ONNX with robust dynamic-batch handling
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

# ---- local path tweak
CUR = os.path.dirname(os.path.abspath(__file__))
if CUR not in sys.path:
    sys.path.insert(0, CUR)

# your project imports (adjust to your repo layout)
from PE.vision_encoder import config as pe_cfg
from PE.vision_encoder.pe import CLIP


def parse_args():
    p = argparse.ArgumentParser(description="Convert PE ckpt to ONNX (vision tower) with dynamic batch")
    p.add_argument("--ckpt", default="assets/FT_PE-Core-L14-336_250804.ckpt", help="Path to ckpt (.ckpt/.pt/.pth)")
    p.add_argument("--onnx", default="assets/model/PE-Core-L14-336_vision_dynamic.onnx", help="Output ONNX path")
    p.add_argument("--model_name", default="PE-Core-L14-336", help="Base model name in config")
    p.add_argument("--split_qkv", action="store_true", help="Use split-qkv variant name suffix")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dynamic_hw", action="store_true", help="Make H/W dynamic too")
    p.add_argument("--no_constant_folding", action="store_true",
                   help="Disable constant folding to avoid baked-in shapes")
    # NEW: post-export reshape fixer
    p.add_argument("--fix_reshape", action="store_true", default=True,
                   help="Post-process ONNX Reshape shapes to copy batch dim (shape[0]=0)")
    return p.parse_args()


def is_vision_only_state_dict(sd_keys):
    s = list(sd_keys)
    has_text_core = any(("token_embedding" in k or "positional_embedding" in k) for k in s)
    mostly_visual = any(k.startswith("visual.") for k in s) or any(
        k.startswith(("conv1.", "transformer.resblocks")) for k in s
    )
    return (not has_text_core) and mostly_visual


class VisualWrapper(nn.Module):
    """(B,3,H,W) -> (B,D)"""
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x):
        return self.visual(x)


def export_onnx(wrapper, dummy, out_path, opset, dynamic_hw, do_const_fold):
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    if dynamic_hw:
        dynamic_axes["input"].update({2: "height", 3: "width"})
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=do_const_fold,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def _set_initializer_array(model, name, np_value):
    from onnx import numpy_helper
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(np_value, name))
            return True
    return False


def _find_const_node(model, name):
    for node in model.graph.node:
        if node.output and node.output[0] == name and node.op_type == "Constant":
            return node
    return None


def _const_node_to_array(node):
    # Constant may carry "value" (TensorProto) or "value_ints"/"value_floats"
    from onnx import numpy_helper
    for attr in node.attribute:
        if attr.name == "value" and attr.t is not None:
            return numpy_helper.to_array(attr.t), "tensor"
        if attr.name == "value_ints" and attr.ints:
            return np.asarray(list(attr.ints), dtype=np.int64), "ints"
        if attr.name == "value_floats" and attr.floats:
            return np.asarray(list(attr.floats), dtype=np.float32), "floats"
    return None, None


def _write_const_node(node, np_value, mode):
    # overwrite node attribute with np_value
    import onnx
    from onnx import helper, numpy_helper
    # remove existing attributes
    del node.attribute[:]
    if mode == "tensor":
        t = numpy_helper.from_array(np.asarray(np_value, dtype=np.int64))
        node.attribute.extend([helper.make_attribute("value", t)])
    elif mode == "ints":
        node.attribute.extend([onnx.helper.make_attribute("value_ints", list(np.asarray(np_value, dtype=np.int64)))])
    elif mode == "floats":
        node.attribute.extend([onnx.helper.make_attribute("value_floats", list(np.asarray(np_value, dtype=np.float32)))])
    else:
        # default to tensor
        t = numpy_helper.from_array(np.asarray(np_value, dtype=np.int64))
        node.attribute.extend([helper.make_attribute("value", t)])


def fix_reshape_batch_dim(onnx_path):
    """
    Post-process: For every Reshape node, if the target shape is a constant/initializer and
    its first dim is 1, set it to 0 (copy batch dim from input).
    This converts patterns like [1, 577, 16, 64] -> [0, 577, 16, 64].
    """
    import onnx
    from onnx import shape_inference

    print(f"[INFO] Post-processing Reshape target shapes in: {onnx_path}")
    model = onnx.load(onnx_path)

    # Fast lookup for initializers by name
    init_map = {init.name: init for init in model.graph.initializer}

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if len(node.input) < 2:
            continue

        data_in, shape_in = node.input[0], node.input[1]

        # Case A: shape is an initializer tensor
        if shape_in in init_map:
            from onnx import numpy_helper
            arr = numpy_helper.to_array(init_map[shape_in]).astype(np.int64, copy=True)
            if arr.ndim == 1 and arr.size >= 1 and arr[0] == 1:
                arr[0] = 0
                _set_initializer_array(model, shape_in, arr)
                changed += 1
                continue

        # Case B: shape comes from a Constant node
        cnode = _find_const_node(model, shape_in)
        if cnode is not None:
            arr, mode = _const_node_to_array(cnode)
            if arr is not None and arr.ndim == 1 and arr.size >= 1 and arr[0] == 1:
                arr = arr.astype(np.int64, copy=True)
                arr[0] = 0
                _write_const_node(cnode, arr, mode)
                changed += 1

    # (Optional) re-run shape inference (safe even with dynamic dims)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] Shape inference after patch failed (ignored): {e}")

    onnx.save(model, onnx_path)
    print(f"[OK] Reshape batch-dim patch applied to {changed} node(s). Saved: {onnx_path}")


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)

    cfg_name = a.model_name + ("-splitqkv" if a.split_qkv else "")
    if cfg_name not in pe_cfg.PE_VISION_CONFIG or cfg_name not in pe_cfg.PE_TEXT_CONFIG:
        raise ValueError(f"Unknown config: {cfg_name} (check vlm_t/vision_encoder/config.py)")

    model = CLIP.from_config(cfg_name, pretrained=False).to(a.device).eval()

    print(f"[INFO] Loading ckpt: {a.ckpt}")
    sd = torch.load(a.ckpt, map_location=a.device)
    if isinstance(sd, dict) and ("state_dict" in sd or "weights" in sd or "model" in sd):
        sd = sd.get("state_dict", sd.get("weights", sd.get("model")))
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # Try full load first
    missing = unexpected = []
    try:
        m, u = model.load_state_dict(sd, strict=False)
        missing, unexpected = list(m), list(u)
        print(f"[INFO] Loaded to CLIP. missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[WARN] load_state_dict to CLIP failed: {e}")

    # If looks vision-only, load into visual submodule
    if isinstance(sd, dict) and (is_vision_only_state_dict(sd.keys()) or len(missing) > 1000):
        print("[INFO] Vision-only checkpoint detected. Loading into visual tower...")
        vis_sd = {}
        for k, v in sd.items():
            if k.startswith("visual."):
                vis_sd[k.replace("visual.", "", 1)] = v
            else:
                vis_sd[k] = v
        m, u = model.visual.load_state_dict(vis_sd, strict=False)
        print(f"[INFO] Visual-only load done. missing={len(m)} unexpected={len(u)}")

    # dummy
    H = W = int(model.image_size)
    dtype = torch.float16 if a.fp16 else torch.float32
    dummy = torch.randn(1, 3, H, W, device=a.device, dtype=dtype)

    wrapper = VisualWrapper(model).to(a.device).eval()

    do_const_fold = not a.no_constant_folding
    print(f"[INFO] Exporting ONNX -> {a.onnx} (opset={a.opset}, const_folding={do_const_fold})")
    with torch.inference_mode():
        export_onnx(wrapper, dummy, a.onnx, a.opset, a.dynamic_hw, do_const_fold)

    # post-process reshape target shapes
    if a.fix_reshape:
        try:
            fix_reshape_batch_dim(a.onnx)
        except Exception as e:
            print(f"[WARN] Reshape patch step failed: {e} (you can re-run with --no_constant_folding)")

    sz_mb = os.path.getsize(a.onnx) / (1024 * 1024)
    print(f"[OK] ONNX saved: {a.onnx} ({sz_mb:.2f} MB)")
    print(f"[INFO] image_size={H}  fp16={a.fp16}  dynamic_hw={a.dynamic_hw}  no_constant_folding={a.no_constant_folding}  fix_reshape={a.fix_reshape}")


if __name__ == "__main__":
    main()
