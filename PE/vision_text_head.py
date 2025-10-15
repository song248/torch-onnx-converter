import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# --- Simple attention head: pools (B, T, D) -> (B, D) with same D ---
class Attention_1(nn.Module):
    def __init__(self, dim: int, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(64, dim // 2)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # scalar score per frame
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: optional Bool (B, T), True=valid, False=padded
        returns: (B, D)
        """
        B, T, D = x.shape
        scores = self.scorer(x).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)          # (B, T)
        v = torch.einsum("bt, btd -> bd", attn, x)   # (B, D)
        return v


class Attention_2(nn.Module):
    """
    Transformer pooling over time + PMA to produce (B, D).
    AMP-safe: returns the SAME dtype as the input x, regardless of internal autocast behavior.
    """
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_pos: bool = False,
        l2_normalize: bool = False,
        init_tau: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_pos = use_pos
        self.l2_normalize = l2_normalize

        # Keep scalar params in fp32 for numerical stability; cast on-the-fly in forward.
        if use_pos:
            self.pos_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        dff = int(dim * mlp_ratio)
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN (stable under AMP)
            )
            for _ in range(depth)
        ])

        # Pooling query kept as fp32 param; runtime cast to x.dtype/device to avoid promotion.
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, dim, dtype=torch.float32) * (dim ** -0.5)
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Optional temperature (not used directly here; kept for API completeness)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau, dtype=torch.float32)))

        # Final norm (usually computes in fp32 under autocast)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, D)  -- may be fp16 under AMP
        mask: Bool (B, T), True=valid (will be inverted for key_padding)
        returns: (B, D) with dtype == x.dtype
        """
        B, T, D = x.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"
        in_dtype = x.dtype
        in_device = x.device

        # --- Lightweight positional cue, cast scale to input dtype to avoid promotion ---
        if self.use_pos:
            pos = torch.linspace(-1.0, 1.0, steps=T, device=in_device, dtype=in_dtype)
            x = x + self.pos_scale.to(dtype=in_dtype, device=in_device) * pos[None, :, None]

        # --- Build key padding mask (True = ignore in PyTorch) ---
        key_padding = None
        if mask is not None:
            # mask: True=valid → invert to True=padded
            key_padding = ~mask
            if key_padding.all(dim=1).any():
                # zero out fully-padded sequences to avoid NaNs
                all_masked = key_padding.all(dim=1)  # (B,)
                if all_masked.any():
                    x = x.clone()
                    x[all_masked] = 0

        # --- Temporal encoder (may compute parts in fp32 under autocast) ---
        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask=key_padding)  # (B, T, D)

        # --- PMA (cast the learned query to the SAME dtype/device as x to prevent upcast) ---
        q = self.pool_query.to(dtype=in_dtype, device=in_device).expand(B, 1, D)  # (B, 1, D)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=key_padding, need_weights=False)  # (B,1,D)
        v = pooled.squeeze(1)  # (B, D)

        # Final norm typically prefers fp32; cast back afterwards.
        v = self.out_norm(v)

        # --- Ensure output dtype matches input dtype (AMP-safe contract) ---
        if v.dtype != in_dtype:
            v = v.to(in_dtype)

        if self.l2_normalize:
            v = F.normalize(v, dim=-1)

        return v




##### Lucas Attentive Probe

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate to
        # [2*lower-1, 2*upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=16,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)


        # --- NEW: merge/squeeze query dimension to get (B, D) ---
        if q.size(1) == 1:
            q = q[:, 0, :]                                   # (B, D)
        else:
            q = q.mean(dim=1)                                # (B, D) if multiple queries

        return q.to(x.dtype)

########################
## Efficient Probe: Attention, Please! Revisiting Attentive Probing for Masked Image Modeling


class EfficientProbing(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_queries = num_queries
        
        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        
        return x_cls



class VideoEfficientPool(nn.Module):
    """
    Pools per-frame features [B, T, C] into a single video vector using EfficientProbing.
    """
    def __init__(self, embed_dim=768, num_queries=32, num_heads=1, d_out=1, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.pool = EfficientProbing(
            dim=embed_dim,
            num_heads=num_heads,      # keep 1 for the given EP code
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            num_queries=num_queries,
            d_out=d_out
        )

    def forward(self, frame_feats, cls=None, attn_mask=None):
        """
        frame_feats: [B, T, C] per-video frame features
        cls: optional external query tokens [B, Q, C] if you want to condition queries
        attn_mask: optional bool mask [B, T] (True = keep / False = pad)  # see note below
        """
        x = frame_feats  # [B, T, C], treat frames as the "tokens"
        # (Optional) If you manage variable T across the batch and have attn_mask,
        # you can adapt EfficientProbing to apply -inf to masked positions before softmax.
        # The provided EP code doesn't include mask handling; see "Masking" section below.
        out = self.pool(x, cls=cls)  # [B, C/d_out]
        return out


##################

class VisionTextHead(nn.Module):
    """
    Wraps a base CLIP-like model to customize video encoding.
    Modes:
      - use=False:    delegate to model.encode_video(frames)
      - vision='linear': residual D->D projection on the default video feature
      - vision='attention': per-frame encode_image + attention pooling (keeps D)
    """
    def __init__(self, args, base_model, device):
        super().__init__()
        self.args = args
        self.base_model = base_model
        device = device


        add_head = self.args.model.additional_head
        head_cfg = getattr(add_head, "vision", None)
        self.use_extra = bool(getattr(add_head, "use", False))
        self.vision_head_type = head_cfg.type
        self.embed_dim   = getattr(head_cfg, "embed_dim", None)
        self.hidden_dim  = getattr(head_cfg, "hidden_dim", None)
        self.dropout     = float(getattr(head_cfg, "dropout", 0.0))
        self.attn_dropout= float(getattr(head_cfg, "attn_dropout", 0.0))
        self.use_pos     = bool(getattr(head_cfg, "use_pos", False))
        self.num_queries = int(getattr(head_cfg, "num_queries", False))
        
        if self.vision_head_type == "attention_1":
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using attention_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using attention_1")
            if self.num_queries:
                raise ValueError("num_queries should be False when using attention_1")
            
            self.vision_head = Attention_1(
                    dim=self.embed_dim,
                    hidden=self.hidden_dim,
                    dropout=self.dropout
                ).to(device)

        # elif self.vision_head_type == "attention_2":
        #     # attention_2 ignores hidden_dim; enforce it is None/0 to avoid config confusion
        #     if self.hidden_dim not in (None, 0):
        #         raise ValueError("hidden_dim must be None or 0 when using attention_2")
        #     if self.num_queries:
        #         raise ValueError("num_queries must be false when using attention_2")

        #     self.vision_head = Attention_2(
        #             dim=self.embed_dim,
        #             dropout=self.dropout,
        #             attn_dropout=self.attn_dropout,
        #             use_pos=self.use_pos
        #         ).to(device)
            
        elif self.vision_head_type == "attentive_1":
            # Lucas head attention
            if self.dropout:
                raise ValueError("dropout should be False when using attentive_1")
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using attentive_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using attentive_1")
            if self.hidden_dim not in (None, 0):
                raise ValueError("hidden_dim must be None or 0 when using attentive_1")
            if self.num_queries:
                raise ValueError("num_queries should be False when using attentive_1")
            
            self.vision_head = AttentivePooler(
                    embed_dim=self.embed_dim,
                    num_heads=16,
                ).to(device)

        elif self.vision_head_type == "vid_efficient_1":
            if self.dropout:
                raise ValueError("dropout should be False when using vid_efficient_1")
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using vid_efficient_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using vid_efficient_1")
            if self.hidden_dim not in (None, 0):
                raise ValueError("hidden_dim must be None or 0 when using vid_efficient_1")

            self.vision_head = VideoEfficientPool(
                    embed_dim=self.embed_dim,
                    num_queries=self.num_queries
                ).to(device)

        else:
            raise ValueError("Wrong self.vision_head_type")
    

    def encode_video(self, video: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if not self.use_extra:
            return self.base_model.encode_video(video, normalize=normalize)

        b, n, c, h, w = video.shape
        frms = video.reshape(b * n, c, h, w)

        frm_feats = self.base_model.encode_image(frms, normalize=False)  # (B*T, D)
        frm_feats = frm_feats.reshape(b, n, -1)                          # (B, T, D)
        # print(f"Frame feat dtype {frm_feats.dtype}")
        video_feats = self.vision_head(frm_feats)                        # (B, D)
        # print(f"video_feats dtype {video_feats.dtype}, shape {video_feats.shape}")

        # keep normalize behavior identical to base path
        if normalize:
            video_feats = F.normalize(video_feats, dim=-1)

        return video_feats

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        text_feats = self.base_model.encode_text(tokens)
        return text_feats

    def logit_scale_exp(self,):
        return self.base_model.logit_scale.exp()
    
    # def save_pretrained_lora(self, save_directory: str):
    #     """
    #     Save only the LoRA adapter weights.
    #     """
    #     self.base_model.save_pretrained(save_directory)