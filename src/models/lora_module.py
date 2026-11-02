"""Low-rank adaptation (LoRA, via `peft`) of a SimMIM-pretrained encoder.

In the manuscript LoRA (rank 2) is applied to the qkv and output projections of the attention
blocks and to both fully-connected layers of the MLPs; the MLP head is trained in full.
"""
from typing import Optional, Sequence

import torch
from lightning import LightningModule
from peft import LoraConfig, get_peft_model

from src.models.components.heads import EncoderWithHead
from src.models.optim import weight_decay_param_groups
from src.models.task_mixins import (
    ClassificationTaskMixin, RegressionTaskMixin, build_classification_loss, plateau_scheduler_config)


class _LoRAModule(LightningModule):

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 output_size: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 lora_config: Optional[LoraConfig] = None,
                 compile: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=['image_encoder'])
        model = EncoderWithHead(image_encoder=image_encoder,
                                output_size=output_size,
                                hidden_layers=hidden_layers)
        if lora_config is None:
            lora_config = LoraConfig(target_modules="all-linear", modules_to_save=["head"])
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
        self.no_weight_decay = image_encoder.no_weight_decay()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # trainable parameters: the LoRA matrices (plus any biases made trainable by the LoRA
        # config) and the trainable copy of the head (`modules_to_save`)
        param_groups = weight_decay_param_groups(self.model.named_parameters(), self.no_weight_decay)
        optimizer = self.hparams.optimizer(params=param_groups, lr=self.hparams.lr, eps=1e-8)
        return plateau_scheduler_config(optimizer, self.hparams.scheduler)


class LoRAClassifierModule(ClassificationTaskMixin, _LoRAModule):
    """LoRA-adapted encoder with a classification head on the CLS token."""

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 lora_config: Optional[LoraConfig] = None,
                 num_classes: int = 2,
                 task: str = 'binary',
                 loss_type: str = 'focal',
                 class_weights: Optional[torch.Tensor] = None,
                 compile: bool = False):
        super().__init__(image_encoder=image_encoder,
                         output_size=num_classes,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
                         hidden_layers=hidden_layers,
                         lora_config=lora_config,
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


class LoRARegressorModule(RegressionTaskMixin, _LoRAModule):
    """LoRA-adapted encoder with a single-output regression head on the CLS token."""

    def __init__(self,
                 image_encoder: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 lr: float,
                 num_epochs: int,
                 hidden_layers: Optional[Sequence[int]] = None,
                 lora_config: Optional[LoraConfig] = None,
                 compile: bool = False):
        super().__init__(image_encoder=image_encoder,
                         output_size=1,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         lr=lr,
                         num_epochs=num_epochs,
                         hidden_layers=hidden_layers,
                         lora_config=lora_config,
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
