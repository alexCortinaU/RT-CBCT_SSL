"""Supervised baselines without pretraining: a randomly initialised ViT and head trained end to end."""
from typing import Optional, Sequence, Tuple

import torch
from lightning import LightningModule

from src.models.components.heads import build_head
from src.models.components.vit import build_vit
from src.models.optim import weight_decay_param_groups
from src.models.task_mixins import (
    ClassificationTaskMixin, RegressionTaskMixin, build_classification_loss, plateau_scheduler_config)


class _SupervisedViTModule(LightningModule):

    def __init__(self,
                 model_name: str,
                 output_size: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 pos_embed_type: str = 'sincos',
                 crop_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: int = 16,
                 compile: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.encoder = build_vit(model_name,
                                 img_size=crop_size,
                                 patch_size=patch_size,
                                 cls_token=True,
                                 pos_embed_type=pos_embed_type)
        self.head = build_head(self.encoder.embed_dim, output_size, hidden_layers)
        self.no_weight_decay = self.encoder.no_weight_decay()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)

    def forward(self, x):
        _, cls_tkn = self.encoder(x, return_cls_token=True)
        return self.head(cls_tkn)

    def configure_optimizers(self):
        named_parameters = list(self.encoder.named_parameters(prefix='encoder')) + \
            list(self.head.named_parameters(prefix='head'))
        param_groups = weight_decay_param_groups(named_parameters, self.no_weight_decay)
        optimizer = self.hparams.optimizer(params=param_groups, lr=self.hparams.lr, eps=1e-8)
        return plateau_scheduler_config(optimizer, self.hparams.scheduler)


class SupervisedClassifierModule(ClassificationTaskMixin, _SupervisedViTModule):
    """Classification from the CLS token of a ViT trained from scratch."""

    def __init__(self,
                 model_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 num_classes: int = 2,
                 task: str = 'binary',
                 loss_type: str = 'focal',
                 class_weights: Optional[torch.Tensor] = None,
                 pos_embed_type: str = 'sincos',
                 crop_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: int = 16,
                 compile: bool = False):
        super().__init__(model_name=model_name,
                         output_size=num_classes,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
                         hidden_layers=hidden_layers,
                         pos_embed_type=pos_embed_type,
                         crop_size=crop_size,
                         patch_size=patch_size,
                         compile=compile)
        self._init_classification_metrics(task, num_classes)
        self.loss = build_classification_loss(loss_type, class_weights)

    def _step(self, stage, batch):
        x, y = batch[:2]
        return self._log_classification_step(stage, self(x), y)

    def training_step(self, batch, batch_idx):
        return self._step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch)

    def test_step(self, batch, batch_idx):
        return self._step('test', batch)


class SupervisedRegressorModule(RegressionTaskMixin, _SupervisedViTModule):
    """Single-target regression from the CLS token of a ViT trained from scratch."""

    def __init__(self,
                 model_name: str,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 pos_embed_type: str = 'sincos',
                 crop_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_size: int = 16,
                 compile: bool = False):
        super().__init__(model_name=model_name,
                         output_size=1,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
                         hidden_layers=hidden_layers,
                         pos_embed_type=pos_embed_type,
                         crop_size=crop_size,
                         patch_size=patch_size,
                         compile=compile)
        self.loss = loss
        self._init_regression_metrics()

    def _step(self, stage, batch):
        x, y = batch[:2]
        return self._log_regression_step(stage, self(x), y)

    def training_step(self, batch, batch_idx):
        return self._step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch)

    def test_step(self, batch, batch_idx):
        return self._step('test', batch)
