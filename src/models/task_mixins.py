"""Loss, metrics and logging shared by all downstream (classification / regression) modules."""
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from monai.losses import FocalLoss
from torch import nn
from torchmetrics import (
    AUROC, Accuracy, ConfusionMatrix, F1Score, MaxMetric, MeanAbsoluteError, PearsonCorrCoef, R2Score)

STAGES = ('train', 'val', 'test')


def build_classification_loss(loss_type: str, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(weight=class_weights, to_onehot_y=True, use_softmax=True)
    raise ValueError(f"Unknown loss_type '{loss_type}', expected 'cross_entropy' or 'focal'")


class ClassificationTaskMixin:
    """Adds accuracy / F1 / AUROC / confusion-matrix tracking for train, val and test.

    For ``task="binary"`` the model outputs two logits and the metrics use the softmax
    probability of class 1.
    """

    def _init_classification_metrics(self, task: str, num_classes: int) -> None:
        self.task = task
        self.num_classes = num_classes
        for stage in STAGES:
            setattr(self, f'acc_{stage}', Accuracy(task=task, num_classes=num_classes))
            setattr(self, f'f1_{stage}', F1Score(task=task, num_classes=num_classes))
            setattr(self, f'auroc_{stage}', AUROC(task=task, num_classes=num_classes))
            setattr(self, f'confmat_{stage}', ConfusionMatrix(task=task, num_classes=num_classes))
        self.max_auroc_val = MaxMetric()

    def _log_classification_step(self, stage: str, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, y)
        self.log(f'{stage}/loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        preds = F.softmax(logits, dim=1)
        getattr(self, f'confmat_{stage}').update(torch.argmax(preds, dim=1), y)
        if self.task == "binary":
            preds = preds[:, 1]
        for name in ('acc', 'f1', 'auroc'):
            metric = getattr(self, f'{name}_{stage}')
            metric(preds, y)
            self.log(f'{stage}/{name}', metric, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _log_confusion_matrix(self, stage: str) -> None:
        confmat = getattr(self, f'confmat_{stage}')
        if isinstance(self.logger, WandbLogger):
            import wandb
            fig, _ = confmat.plot()
            self.logger.experiment.log({f'{stage}/confmat': [wandb.Image(fig)]})
            plt.close(fig)
        confmat.reset()

    def on_train_epoch_end(self):
        self._log_confusion_matrix('train')

    def on_validation_epoch_end(self):
        self._log_confusion_matrix('val')
        if not self.trainer.sanity_checking:
            self.max_auroc_val.update(self.auroc_val.compute())
            self.log('val/max_auroc', self.max_auroc_val.compute())

    def on_test_epoch_end(self):
        self._log_confusion_matrix('test')


class RegressionTaskMixin:
    """Adds R2 / Pearson / MAE tracking for train, val and test (single regression target)."""

    def _init_regression_metrics(self) -> None:
        for stage in STAGES:
            setattr(self, f'r2_{stage}', R2Score())
            setattr(self, f'pearson_{stage}', PearsonCorrCoef())
            setattr(self, f'mae_{stage}', MeanAbsoluteError())

    def _log_regression_step(self, stage: str, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss(preds, y)
        self.log(f'{stage}/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        for name in ('r2', 'pearson', 'mae'):
            metric = getattr(self, f'{name}_{stage}')
            metric(preds.squeeze(), y.squeeze())
            self.log(f'{stage}/{name}', metric, on_step=False, on_epoch=True, sync_dist=True)
        return loss


def plateau_scheduler_config(optimizer, scheduler):
    """Return the Lightning optimizer config; the scheduler (if any) steps on ``val/loss_epoch``."""
    if not scheduler:
        return {'optimizer': optimizer}
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler(optimizer=optimizer),
            'monitor': 'val/loss_epoch',
            'interval': 'epoch',
            'frequency': 1,
        }
    }
