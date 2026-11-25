"""SimMIM masked image modeling for 3D CBCT volumes (Xie et al., 2022).

A ViT encoder sees the volume with ~60% of its 32^3-voxel mask patches replaced by a mask
token; a 1x1x1 convolution decodes every token back to its 16^3 voxels and the L1 loss is
computed on the masked voxels only.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.models.components.vit import build_vit
from src.models.optim import weight_decay_param_groups
from src.utils.tensors import trunc_normal_


class SimMIMModule(LightningModule):
    """SimMIM pretraining module.

    :param model_name: ``vit_tiny``, ``vit_small`` or ``vit_base``.
    :param optimizer: Partially-initialised optimizer (params and lr are set here).
    :param scheduler: Optional partially-initialised LR scheduler (stepped on ``val/loss_epoch``).
    :param num_epochs: Number of pretraining epochs (read by the trainer config).
    :param cls_token: Prepend a CLS token to the encoder input (needed downstream).
    :param normalize_output: Apply tanh to the reconstruction so it matches the [-1, 1] input range.
    :param pos_embed_type: ``sincos`` or ``learnable``.
    :param crop_size: Input volume size.
    :param patch_size: ViT patch size.
    """

    def __init__(self,
                 model_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 num_epochs: int,
                 compile: bool = True,
                 cls_token: bool = False,
                 normalize_output: bool = False,
                 pos_embed_type: str = 'learnable',
                 lr: float = 1e-4,
                 crop_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: int = 16):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.encoder = build_vit(
            model_name,
            img_size=crop_size,
            patch_size=patch_size,
            cls_token=cls_token,
            pos_embed_type=pos_embed_type)
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        # lightweight decoder: one linear projection per token back to its patch voxels
        self.decoder = torch.nn.Conv3d(
            in_channels=self.encoder.embed_dim,
            out_channels=self.patch_size ** 3,
            kernel_size=1)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

        for m in self.encoder.modules():
            init_weights(m)
        for m in self.decoder.modules():
            init_weights(m)

        self.no_weight_decay = self.encoder.no_weight_decay()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, avgpool=True)

    def reconstruct(self, imgs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full volume from the masked input."""
        z = self.encoder(imgs, masks)
        x_rec = self.decoder(z).reshape(z.shape[0], -1, *self.hparams.crop_size)
        if self.hparams.normalize_output:
            x_rec = torch.tanh(x_rec)
        return x_rec

    def masked_l1_loss(self, imgs: torch.Tensor, x_rec: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """L1 reconstruction loss averaged over the masked voxels only."""
        masks = self.repeat_mask(masks, self.patch_size)
        loss_recon = F.l1_loss(imgs, x_rec, reduction='none')
        return (loss_recon * masks).sum() / (masks.sum() + 1e-5) / self.in_chans

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        imgs, masks = batch
        loss = self.masked_l1_loss(imgs, self.reconstruct(imgs, masks), masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        imgs, masks = batch
        loss = self.masked_l1_loss(imgs, self.reconstruct(imgs, masks), masks)
        self.log('val/loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        named_parameters = list(self.encoder.named_parameters()) + list(self.decoder.named_parameters())
        param_groups = weight_decay_param_groups(named_parameters, self.no_weight_decay)
        optimizer = self.hparams.optimizer(params=param_groups, lr=self.hparams.lr)
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_epoch",
                "interval": "epoch",
                "frequency": 1
            },
        }

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def repeat_mask(mask: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Upsample a token-level mask (B, h, w, d) to voxel level (B, 1, H, W, D)."""
        mask = torch.repeat_interleave(mask, num_repeats, dim=1)
        mask = torch.repeat_interleave(mask, num_repeats, dim=2)
        mask = torch.repeat_interleave(mask, num_repeats, dim=3)
        return mask.unsqueeze(1).contiguous()
