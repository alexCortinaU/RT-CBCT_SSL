# Adapted from I-JEPA (https://github.com/facebookresearch/ijepa) and
# SimMIM (https://github.com/microsoft/SimMIM).
# Copyright (c) Meta Platforms, Inc. and affiliates. / Copyright (c) 2021 Microsoft.
# Licensed under the respective licenses of the original repositories.
"""3D Vision Transformer encoder used for SimMIM pretraining and all downstream tasks.

The encoder tokenises a 3D volume with a 3D convolutional patch embedding, adds fixed
sin-cos (or learnable) positional embeddings, optionally prepends a CLS token, and
replaces masked tokens with a learnable mask token during SimMIM pretraining.
"""
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from timm.layers import trunc_normal_ as timm_trunc_normal_

from src.utils.tensors import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
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
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """3D image to patch embedding."""

    def __init__(self, img_size=(384, 384, 64), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_shape = tuple(int(dim // patch_size) for dim in img_size)
        self.patch_size = patch_size
        self.num_patches = int(np.prod(self.patch_shape))
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # BCHWD -> BNC


class FeatAvgPool(nn.Module):
    """Average the patch tokens: (B, N, C) -> (B, C)."""

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        return self.avg_pool(x).squeeze()


class VisionTransformer3D(nn.Module):
    """3D ViT encoder with an optional CLS token and a SimMIM mask token.

    :param img_size: Input volume size (H, W, D); every dimension must be divisible by `patch_size`.
    :param patch_size: Side length of the cubic patches (tokens).
    :param cls_token: Prepend a learnable CLS token (used as the image embedding downstream).
    :param pos_embed_type: ``"sincos"`` (fixed, used in the manuscript) or ``"learnable"``.
    """

    def __init__(
        self,
        img_size=(224, 224, 64),
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        pos_embed_type='sincos',
        cls_token=False,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if pos_embed_type == 'sincos':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                          requires_grad=False)
            pos_embed = build_sincos_position_embedding(grid_size=list(self.patch_embed.patch_shape),
                                                        embed_dim=embed_dim,
                                                        spatial_dims=3)
            self.pos_embed.data.copy_(pos_embed.float())
        elif pos_embed_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=init_std, a=-init_std, b=init_std)
        else:
            raise NotImplementedError(f"Position embedding type {pos_embed_type} not implemented")
        self.pos_embed_type = pos_embed_type

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

        # SimMIM-specific tokens (created after the backbone so that seeded initialisations
        # match the checkpoints released with the manuscript)
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        timm_trunc_normal_(self.mask_token, mean=0., std=.02, a=-.02, b=.02)
        if cls_token:
            timm_trunc_normal_(self.cls_token, mean=0., std=.02, a=-.02, b=.02)
        self.avg_pool = FeatAvgPool()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self,
                x,
                mask=None,
                avgpool=False,
                return_embds=False,
                return_cls_token=False):
        """Encode a batch of volumes.

        :param x: Input of shape (B, C, H, W, D).
        :param mask: Optional SimMIM mask of shape (B, H/p, W/p, D/p); 1 = masked token.
        :param avgpool: Return the average of the patch tokens, shape (B, C').
        :param return_embds: Return the patch tokens, shape (B, N, C').
        :param return_cls_token: Additionally return the CLS token, shape (B, C').
        :return: By default the patch tokens as a feature map of shape (B, C', H/p, W/p, D/p).
        """
        x = self.patch_embed(x)
        B, L, _ = x.shape

        if mask is not None:
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(f"Input has {x.shape[1]} patches but the positional embedding was built "
                             f"for {self.pos_embed.shape[1]}; check that the input size matches img_size.")
        x = x + self.pos_embed

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if hasattr(self, "cls_token"):
            x_cls_token = x[:, 0]
            x = x[:, 1:]

        B, L, C = x.shape
        H, W, D = self.patch_embed.patch_shape

        if return_embds:
            return x
        if avgpool:
            return self.avg_pool(x)

        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)

        if return_cls_token and hasattr(self, "cls_token"):
            return x, x_cls_token
        return x


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer3D(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer3D(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer3D(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


VIT_MODELS = {
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_large': vit_large,
}


def build_vit(model_name: str, **kwargs) -> VisionTransformer3D:
    """Build a ViT by name (``vit_tiny``, ``vit_small``, ``vit_base`` or ``vit_large``)."""
    if model_name not in VIT_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}', expected one of {list(VIT_MODELS)}")
    return VIT_MODELS[model_name](**kwargs)
