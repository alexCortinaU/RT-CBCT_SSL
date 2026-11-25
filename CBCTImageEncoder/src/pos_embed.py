"""3D sin-cos position embedding, vendored to avoid depending on the full `monai`
library for one ~30-line function. Numerically identical to
`monai.networks.blocks.pos_embed_utils.build_sincos_position_embedding(spatial_dims=3)`,
which this model was originally built and trained with.

Reference: https://github.com/cvlab-stonybrook/SelfMedMAE
"""

import torch


def build_sincos_position_embedding(
    grid_size: tuple[int, int, int], embed_dim: int, temperature: float = 10000.0
) -> torch.nn.Parameter:
    """Fixed (non-learnable) 3D sin-cos position embedding for a `grid_size` patch grid.

    `embed_dim` must be divisible by 6 (2 sin/cos pairs per spatial axis).
    """
    if embed_dim % 6 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 6 for 3D sin-cos position embedding")

    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)
    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing="ij")

    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_d = torch.einsum("m,d->md", grid_d.flatten(), omega)

    pos_emb = torch.cat(
        [
            torch.sin(out_w), torch.cos(out_w),
            torch.sin(out_h), torch.cos(out_h),
            torch.sin(out_d), torch.cos(out_d),
        ],
        dim=1,
    )[None, :, :]

    pos_embed = torch.nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed
