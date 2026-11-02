import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class LoraCheckpoint(Callback):
    """Save the LoRA adapters (and head) of the best epoch with ``peft``'s ``save_pretrained``.

    The adapters are written to ``<exp_path>/<dir_name>/lora_best`` whenever ``monitor`` improves
    (the same criterion as the ``ModelCheckpoint`` callback), together with ``best_epoch.json``.
    """

    def __init__(self, exp_path: str = './', dir_name: str = 'lora_checkpoints', monitor: str = 'val/auroc',
                 mode: str = 'max'):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.dirpath = Path(exp_path) / dir_name
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score > self.best_score if self.mode == 'max' else score < self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.monitor not in trainer.callback_metrics:
            return
        score = float(trainer.callback_metrics[self.monitor])
        if self._improved(score):
            self.best_score = score
            best_dir = self.dirpath / 'lora_best'
            pl_module.model.save_pretrained(best_dir)
            with open(best_dir / 'best_epoch.json', 'w') as f:
                json.dump({'epoch': trainer.current_epoch, 'monitor': self.monitor, 'score': score}, f, indent=4)


class SaveSplitSeed(Callback):
    """Record the training subset (fraction, seed and rows) used in a finetuning run."""

    def __init__(self, dirpath: str = './'):
        self.dirpath = Path(dirpath)

    def on_fit_start(self, trainer, pl_module):
        data_train = trainer.datamodule.data_train
        split_info = {
            'train_subsplit': data_train.train_subsplit if data_train.train_subsplit else 'Not found',
            'seed': data_train.seed if data_train.seed else 'Not found',
        }
        with open(self.dirpath / 'data_split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)
        data_train.df.to_csv(self.dirpath / 'data_split_df.csv', index=False)


class PlotReconstructions(Callback):
    """Plot SimMIM reconstructions of a few validation volumes (three orthogonal mid-slices).

    The figure is logged to Weights & Biases when a ``WandbLogger`` is used and saved as png
    in ``dirpath`` otherwise.
    """

    def __init__(self, plot_every: int = 1, dirpath: str = None):
        self.plot_every = plot_every
        self.dirpath = Path(dirpath) if dirpath else None
        self.imgs = self.masks = None

    def on_fit_start(self, trainer, pl_module):
        self.imgs, self.masks = next(iter(trainer.datamodule.examples_dataloader()))

    @staticmethod
    def plot_3views(imgs, masks, recons, patch_size=16):
        fig = plt.figure(figsize=(12, 10))
        outer = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
        for i, (img, mask, recon) in enumerate(zip(imgs, masks, recons)):
            # show the input with the masked patches replaced by their reconstruction
            mask = np.repeat(np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1), patch_size, axis=2)
            img = img.clone()
            img[0][mask == 1] = recon[0][mask == 1]

            inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            views = [(img[0, img.shape[1] // 2, ...], 2.5),
                     (img[0, :, img.shape[2] // 2, :], 2.5),
                     (img[0, ..., img.shape[3] // 2], 1)]
            for j, (view, aspect) in enumerate(views):
                ax = plt.Subplot(fig, inner[j])
                ax.imshow(torch.rot90(view), cmap='gray', aspect=aspect)
                ax.axis('off')
                fig.add_subplot(ax)
        return fig

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch % self.plot_every != 0:
            return
        with torch.no_grad():
            x_rec = pl_module.reconstruct(self.imgs.to(pl_module.device), self.masks.to(pl_module.device))
        fig = self.plot_3views(self.imgs, self.masks, x_rec.cpu(), patch_size=pl_module.patch_size)

        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log({"reconstructions": [wandb.Image(fig)]})
        elif self.dirpath is not None:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.dirpath / f'reconstructions_epoch_{trainer.current_epoch:03d}.png')
        plt.close(fig)
