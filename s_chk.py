import torch, re

ckpt_path = "assets/FT_PE-Core-L14-336_250804.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")
if isinstance(sd, dict):
    sd = sd.get("state_dict", sd.get("weights", sd.get("model", sd)))

keys = list(sd.keys())

# 1) 키 이름으로 빠른 스캔
has_split_names = any(re.search(r'(?:^|\.)(q_proj|k_proj|v_proj)\.(weight|bias)$', k) for k in keys)
has_fused_names = any(re.search(r'(?:^|\.)(?:in_proj_weight|in_proj_bias|qkv\.weight|qkv\.bias)$', k) for k in keys)

print("[scan] split-style keys:", has_split_names)
print("[scan] fused qkv keys :", has_fused_names)

# 2) shape로도 확인 (임의로 몇 개만)
def first(tnames):
    for t in tnames:
        for k in keys:
            if k.endswith(t):
                return k, sd[k].shape
    return None, None

k_fused, shp_fused = first(["in_proj_weight", "qkv.weight"])
k_q, shp_q = first(["q_proj.weight"])
k_k, shp_k = first(["k_proj.weight"])
k_v, shp_v = first(["v_proj.weight"])

print("fused example:", k_fused, shp_fused)
print("q_proj example:", k_q, shp_q)
print("k_proj example:", k_k, shp_k)
print("v_proj example:", k_v, shp_v)

# 3) 판정 로직
is_split_by_names = has_split_names and not has_fused_names
is_fused_by_names = has_fused_names and not has_split_names

is_split_by_shapes = (shp_q is not None and shp_k is not None and shp_v is not None
                      and len(shp_q) == 2 and shp_q[0] == shp_k[0] == shp_v[0] and shp_q[0] == shp_q[1])
is_fused_by_shapes = (shp_fused is not None and len(shp_fused) == 2 and shp_fused[0] % 3 == 0)

if is_split_by_names or is_split_by_shapes:
    print("=> 추정: split-qkv 로 학습된 ckpt")
elif is_fused_by_names or is_fused_by_shapes:
    print("=> 추정: qkv 통합(fused) 구조의 ckpt")
else:
    print("=> 확정 불가: 키 접두사/shape가 애매합니다. (아래 방법 B 권장)")
