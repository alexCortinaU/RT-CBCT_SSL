"""Test-set inference shared by the scripts in ``src/inference``."""
import json
from pathlib import Path
from typing import Any, Dict

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.metrics import compute_classification_metrics, compute_regression_metrics
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def run_inference(cfg: DictConfig, model: torch.nn.Module, metadata: Dict[str, Any]) -> Dict[str, float]:
    """Run ``model`` on the test set of ``cfg.data`` and save metrics and predictions.

    Writes to the run output directory: ``test_metrics.json``, ``test_predictions.pt`` (logits),
    ``test_targets.pt``, ``metadata.json`` and, if the datamodule returns them,
    ``test_image_paths.json``.
    """
    device = torch.device(f"cuda:{cfg.get('device', 0)}" if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device {device}')
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    log.info(f"Test dataset of length: {len(datamodule.data_test)}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    all_outputs, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for batch in tqdm(datamodule.test_dataloader(), desc='Running inference on test set'):
            images, targets = batch[0].to(device), batch[1]
            if len(batch) > 2:
                all_paths.extend(batch[2])
            all_outputs.append(model(images).detach().cpu())
            all_targets.append(targets.detach().cpu())

    if cfg.model.get('task'):
        metrics = compute_classification_metrics(all_outputs, all_targets,
                                                 task=cfg.model.task, num_classes=cfg.model.num_classes)
    else:
        metrics = compute_regression_metrics(all_outputs, all_targets)
    log.info(f'Performance metrics on test set: {metrics}')

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    torch.save(torch.cat(all_outputs, dim=0), output_dir / 'test_predictions.pt')
    torch.save(torch.cat(all_targets, dim=0), output_dir / 'test_targets.pt')
    if all_paths:
        with open(output_dir / 'test_image_paths.json', 'w') as f:
            json.dump(all_paths, f, indent=4)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({'experiment_name': cfg.get('exp_name'), 'output_dir': str(output_dir), **metadata}, f, indent=4)
    log.info(f'Saved test metrics and predictions to {output_dir}')
    return metrics
