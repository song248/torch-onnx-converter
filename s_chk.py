import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

# Optional imports guarded
try:
    import onnx
    from onnx import shape_inference
except Exception as e:
    print("[WARN] onnx package not available. Install with: pip install onnx")
    onnx = None
    shape_inference = None

try:
    import onnxruntime as ort
except Exception as e:
    print("[WARN] onnxruntime not available. Install with: pip install onnxruntime-gpu "
          "or pip install onnxruntime")
    ort = None


def load_io_shapes(onnx_path: str) -> Tuple[List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]:
    """Read model input/output tensor names and shapes (with -1 for dynamic)."""
    if onnx is None:
        raise RuntimeError("onnx package is required for static shape inspection.")

    model = onnx.load(onnx_path)
    graph = model.graph

    def dims_to_list(t):
        shape = []
        for d in t.type.tensor_type.shape.dim:
            if d.dim_param:  # symbolic, e.g., "batch" / "height"
                shape.append(-1)
            elif d.dim_value is not None and d.dim_value != 0:
                shape.append(int(d.dim_value))
            else:
                shape.append(-1)
        return shape

    inputs = [(vi.name, dims_to_list(vi)) for vi in graph.input]
    outputs = [(vi.name, dims_to_list(vi)) for vi in graph.output]
    return inputs, outputs


def try_shape_inference(onnx_path: str) -> bool:
    """Run ONNX shape inference to ensure the graph is sane (optional but useful)."""
    if onnx is None or shape_inference is None:
        print("[INFO] Skip shape inference (onnx not available).")
        return True
    try:
        _ = shape_inference.infer_shapes(onnx.load(onnx_path))
        print("[OK] ONNX shape inference passed.")
        return True
    except Exception as e:
        print("[FAIL] ONNX shape inference failed:", str(e))
        return False


def pick_providers(prefer_cuda: bool = True) -> List[str]:
    if ort is None:
        return []
    available = ort.get_available_providers()
    providers = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def run_ort_batches(onnx_path: str, img_size: int, batches: List[int]) -> bool:
    if ort is None:
        print("[INFO] Skip runtime check (onnxruntime not available).")
        return True

    providers = pick_providers(prefer_cuda=True)
    if not providers:
        print("[FAIL] No ORT providers available. Install onnxruntime or onnxruntime-gpu.")
        return False

    print(f"[INFO] ORT providers: {providers}")
    so = ort.SessionOptions()
    so.log_severity_level = 2  # reduce verbosity
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    io_bindings = {
        "inputs": [i.name for i in sess.get_inputs()],
        "outputs": [o.name for o in sess.get_outputs()],
    }
    if len(io_bindings["inputs"]) != 1 or len(io_bindings["outputs"]) != 1:
        print(f"[WARN] This checker assumes 1 input & 1 output. "
              f"Found inputs={io_bindings['inputs']}, outputs={io_bindings['outputs']}")

    in_name = io_bindings["inputs"][0]
    out_name = io_bindings["outputs"][0]

    ok_all = True
    for b in batches:
        x = np.random.rand(b, 3, img_size, img_size).astype(np.float32)
        try:
            out = sess.run([out_name], {in_name: x})[0]
            print(f"[OK] ORT inference batch={b} -> output shape={tuple(out.shape)}")
        except Exception as e:
            print(f"[FAIL] ORT inference failed for batch={b}: {e}")
            ok_all = False
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="Check ONNX dynamic batch behavior")
    ap.add_argument("--onnx", default = "assets/model/PE-Core-L14-336_vision_dynamic.onnx")
    ap.add_argument("--img-size", type=int, default=336, help="Input H=W size")
    ap.add_argument("--batches", default="1,2,4,8,16",
                    help="Comma-separated batch sizes to test, e.g. 1,2,4")
    args = ap.parse_args()

    onnx_path = args.onnx
    if not os.path.exists(onnx_path):
        print(f"[ERR] ONNX not found: {onnx_path}")
        sys.exit(1)

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    print(f"[INFO] Checking: {onnx_path}")
    print(f"[INFO] Test batches: {batches}  (img-size={args.img_size})")

    # 1) Static IO shape check
    try:
        inputs, outputs = load_io_shapes(onnx_path)
        print("== IO (static inspection) ==")
        for n, s in inputs:
            print(f"Input  : {n:>20}  shape={s}")
        for n, s in outputs:
            print(f"Output : {n:>20}  shape={s}")

        dyn_ok = False
        # Expect first dim of at least one input & output to be -1
        if inputs and outputs:
            dyn_ok = (inputs[0][1][0] == -1) and (outputs[0][1][0] == -1)
        if dyn_ok:
            print("[OK] First dim (batch) is dynamic (-1) for input & output.")
        else:
            print("[WARN] Batch dim does not appear dynamic in IO shapes. "
                  "Graph may still be partially dynamic; proceed to runtime test.")
    except Exception as e:
        print("[WARN] Static IO check skipped due to error:", e)

    # 2) ONNX shape inference (sanity)
    infer_ok = try_shape_inference(onnx_path)

    # 3) Runtime test with ORT
    ort_ok = run_ort_batches(onnx_path, img_size=args.img_size, batches=batches)

    # Summary
    print("\n== SUMMARY ==")
    print(f"Shape inference  : {'PASS' if infer_ok else 'FAIL'}")
    print(f"ORT runtime test : {'PASS' if ort_ok else 'FAIL'}")

    if not ort_ok:
        print("\n[Hint] If certain batches fail:")
        print("- ONNX 내부에 배치=1 가정 Reshape가 남아있을 수 있습니다.")
        print("- export 시 do_constant_folding=False, output dynamic_axes 포함 확인.")
        print("- 모델 내부 view/reshape에서 배치 차원을 입력에서 유도(x.size(0))하도록 수정하세요.")

    if infer_ok and ort_ok:
        print("\n✅ Dynamic batch ONNX looks good!")


if __name__ == "__main__":
    main()
