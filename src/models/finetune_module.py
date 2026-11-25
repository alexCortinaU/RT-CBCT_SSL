"""Downstream heads on top of a SimMIM-pretrained encoder (full finetuning and linear probing).

The head receives the CLS token of the encoder. With ``freeze_encoder=True`` only the head is
trained (linear probing when ``hidden_layers`` is empty); with ``freeze_encoder=False`` all
encoder weights are finetuned together with the head (full finetuning).
"""
from typing import Optional, Sequence

import torch
from lightning import LightningModule

from src.models.components.heads import build_head
from src.models.optim import weight_decay_param_groups
from src.models.task_mixins import (
    ClassificationTaskMixin, RegressionTaskMixin, build_classification_loss, plateau_scheduler_config)


class _PretrainedEncoderModule(LightningModule):
    """Pretrained encoder + head on the CLS token, optionally with a frozen encoder."""

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 output_size: int,
                 hidden_layers: Optional[Sequence[int]],
                 freeze_encoder: bool,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 compile: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=['image_encoder'])
        self.image_encoder = image_encoder
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self.head = build_head(image_encoder.embed_dim, output_size, hidden_layers)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.image_encoder = torch.compile(self.image_encoder)

    def forward_encoder(self, x):
        if self.freeze_encoder:
            self.image_encoder.eval()
            with torch.no_grad():
                _, cls_tkn = self.image_encoder(x, return_cls_token=True)
        else:
            _, cls_tkn = self.image_encoder(x, return_cls_token=True)
        return cls_tkn

    def forward(self, x):
        return self.head(self.forward_encoder(x))

    def configure_optimizers(self):
        # frozen encoder parameters (linear probing) are skipped by the helper
        named_parameters = list(self.image_encoder.named_parameters(prefix='image_encoder')) + \
            list(self.head.named_parameters(prefix='head'))
        param_groups = weight_decay_param_groups(named_parameters, self.image_encoder.no_weight_decay())
        optimizer = self.hparams.optimizer(params=param_groups, lr=self.hparams.lr, eps=1e-8)
        return plateau_scheduler_config(optimizer, self.hparams.scheduler)


class ClassifierModule(ClassificationTaskMixin, _PretrainedEncoderModule):
    """Classification (binary or multiclass) from the CLS token of a pretrained encoder.

    :param num_classes: Number of classes (2 for binary tasks, which use two logits).
    :param task: ``binary`` or ``multiclass`` (torchmetrics task).
    :param loss_type: ``focal`` (used in the manuscript) or ``cross_entropy``.
    :param class_weights: Optional per-class loss weights (computed by the datamodule).
    """

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 freeze_encoder: bool,
                 hidden_layers: Optional[Sequence[int]] = None,
                 num_classes: int = 2,
                 task: str = 'binary',
                 loss_type: str = 'focal',
                 class_weights: Optional[torch.Tensor] = None,
                 compile: bool = False):
        super().__init__(image_encoder=image_encoder,
                         output_size=num_classes,
                         hidden_layers=hidden_layers,
                         freeze_encoder=freeze_encoder,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
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


class RegressorModule(RegressionTaskMixin, _PretrainedEncoderModule):
    """Single-target regression from the CLS token of a pretrained encoder.

    :param loss: Loss module, e.g. ``torch.nn.MSELoss()``.
    """

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 freeze_encoder: bool,
                 hidden_layers: Optional[Sequence[int]] = None,
                 compile: bool = False):
        super().__init__(image_encoder=image_encoder,
                         output_size=1,
                         hidden_layers=hidden_layers,
                         freeze_encoder=freeze_encoder,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
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
