"""Test-set metrics computed by the inference scripts from the stacked model outputs."""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, MeanAbsoluteError, PearsonCorrCoef, R2Score
from torchmetrics.classification import MulticlassAUROC

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def compute_classification_metrics(all_outputs: List[torch.Tensor],
                                   all_targets: List[torch.Tensor],
                                   task: str = 'binary',
                                   num_classes: int = 2) -> Dict[str, float]:
    """Classification metrics from the logits of a model.

    For multiclass tasks ``macro_present_auroc`` is the one-vs-rest AUROC averaged over the
    classes present in the test set (the manuscript excludes T0 and N3, which have no test cases).

    :param all_outputs: List of logit tensors of shape (B, num_classes).
    :param all_targets: List of integer label tensors of shape (B,).
    """
    logits = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = F.softmax(logits, dim=1)
    preds = probs[:, 1] if task == 'binary' else probs

    metrics = {
        'accuracy': float(Accuracy(task=task, num_classes=num_classes)(preds, targets)),
        'f1_score': float(F1Score(task=task, num_classes=num_classes)(preds, targets)),
        'auc': float(AUROC(task=task, num_classes=num_classes)(preds, targets)),
    }

    present_classes = np.unique(targets.numpy()).astype(int)
    if len(present_classes) != num_classes:
        log.warning(f'Only classes {present_classes.tolist()} are present in the targets '
                    f'(num_classes={num_classes}); roc_auc_sklearn is set to NaN.')
        metrics['roc_auc_sklearn'] = float('nan')
    else:
        metrics['roc_auc_sklearn'] = float(roc_auc_score(
            y_true=targets.numpy(),
            y_score=probs.numpy() if task == 'multiclass' else probs[:, 1].numpy(),
            multi_class='ovr' if task == 'multiclass' else 'raise'))

    cm = ConfusionMatrix(task=task, num_classes=num_classes)(torch.argmax(probs, dim=1), targets).numpy()
    if task == 'binary':
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        metrics.update({
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(tp / (tp + fp)),
            'npv': float(tn / (tn + fn)),
            'balanced_accuracy': float((sensitivity + specificity) / 2),
        })
    else:
        per_class_auroc = MulticlassAUROC(num_classes=num_classes, average=None)(probs, targets)
        present = torch.bincount(targets, minlength=num_classes) > 0
        per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        metrics.update({
            'confusion_matrix': cm.tolist(),
            'per_class_auroc': per_class_auroc.tolist(),
            'macro_present_auroc': float(per_class_auroc[present].mean()),
            'balanced_accuracy': float(per_class_recall[present.numpy()].mean()),
        })
    return metrics


def compute_regression_metrics(all_outputs: List[torch.Tensor],
                               all_targets: List[torch.Tensor]) -> Dict[str, float]:
    """MAE, Pearson correlation and R2 (in the normalised target units)."""
    outputs = torch.cat(all_outputs, dim=0).squeeze()
    targets = torch.cat(all_targets, dim=0).squeeze()
    return {
        'mae': float(MeanAbsoluteError()(outputs, targets)),
        'pearson': float(PearsonCorrCoef()(outputs, targets)),
        'r2': float(R2Score()(outputs, targets)),
    }
