"""A 3D Vision Transformer for CBCT volumes, and a loader for our pretrained checkpoints.

This is a clean, minimal rewrite of the encoder actually used to produce the released
weights -- not a literal copy of the full research codebase. Dropped, relative to the
original: SimMIM's `mask_token` and masking machinery (training-only, irrelevant once
you're just extracting embeddings), unused dead code paths, and the `monai`/`timm`
dependencies (see `pos_embed.py`).
"""

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .pos_embed import build_sincos_position_embedding

_INIT_STD = 0.02


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_shape = tuple(dim // patch_size for dim in img_size)
        self.num_patches = self.patch_shape[0] * self.patch_shape[1] * self.patch_shape[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B,C,H,W,D) -> (B,N,C)


class VisionTransformer3D(nn.Module):
    """3D ViT encoder with a CLS token. Call `forward(x, return_cls_token=True)` to get
    both the patch-token grid and the CLS embedding; omit it for just the patch tokens.
    """

    def __init__(
        self,
        img_size=(256, 224, 64),
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.img_size = tuple(img_size)
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = build_sincos_position_embedding(self.patch_embed.patch_shape, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=_INIT_STD)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=_INIT_STD)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_cls_token: bool = False):
        """x: (B, in_chans, H, W, D) volume, already preprocessed to `self.img_size`.

        Default: returns all patch tokens, (B, N_patches, embed_dim).
        `return_cls_token=True`: returns (patch_tokens, cls_token), cls shape (B, embed_dim).
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed  # pos_embed covers patch tokens only, added before CLS is prepended
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_out, patch_tokens = x[:, 0], x[:, 1:]
        if return_cls_token:
            return patch_tokens, cls_out
        return patch_tokens


_ARCH = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
}


def build_model(model_size: str, img_size=(256, 224, 64), patch_size: int = 16, in_chans: int = 1) -> VisionTransformer3D:
    """Instantiate an (untrained) encoder of the given size: 'tiny', 'small', or 'base'."""
    if model_size not in _ARCH:
        raise ValueError(f"model_size must be one of {list(_ARCH)}, got {model_size!r}")
    return VisionTransformer3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, **_ARCH[model_size])


def load_pretrained(model_size: str, checkpoints_dir: str = "checkpoints") -> VisionTransformer3D:
    """Load a pretrained encoder from `<checkpoints_dir>/vit_<model_size>_{config.yaml,weights.ckpt}`.

    `vit_<size>_config.yaml` supplies the architecture args (img_size/patch_size/in_chans);
    `vit_<size>_weights.ckpt` is a plain state_dict (not a Lightning checkpoint).
    """
    ckpt_dir = Path(checkpoints_dir)
    config_path = ckpt_dir / f"vit_{model_size}_config.yaml"
    weights_path = ckpt_dir / f"vit_{model_size}_weights.ckpt"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found -- weights are distributed separately, see checkpoints/README.md"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    expected_embed_dim = _ARCH[model_size]["embed_dim"]
    if config.get("embed_dim", expected_embed_dim) != expected_embed_dim:
        raise ValueError(
            f"{config_path} has embed_dim={config['embed_dim']}, but model_size={model_size!r} "
            f"expects {expected_embed_dim} -- config/weights mismatch."
        )

    model = build_model(
        model_size,
        img_size=tuple(config["img_size"]),
        patch_size=config["patch_size"],
        in_chans=config.get("in_chans", 1),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
