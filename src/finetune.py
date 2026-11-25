"""Finetune a pretrained encoder with a head on the CLS token: full finetuning or linear probing.

Full finetuning (all encoder weights are trained, ``model.freeze_encoder=false``):
    python src/finetune.py experiment=feeding_tube/full_finetune model_name=vit-s

Linear probing of a pretraining checkpoint (frozen encoder + linear layer):
    python src/finetune.py experiment=feeding_tube/linear_probe model_name=vit-s pretrained_epoch=10
"""
from typing import Optional

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import extras, get_metric_value, task_wrapper  # noqa: E402
from src.utils.checkpoints import load_pretrained_encoder  # noqa: E402
from src.utils.training import class_weights_kwargs, train_and_test  # noqa: E402


def build_model(cfg: DictConfig, datamodule: LightningDataModule) -> LightningModule:
    image_encoder = load_pretrained_encoder(cfg.pretrained_ckpt)
    return hydra.utils.instantiate(cfg.model, image_encoder=image_encoder, **class_weights_kwargs(datamodule))


@task_wrapper
def finetune(cfg: DictConfig):
    # the best checkpoint (see callbacks.model_checkpoint) is evaluated on the test set
    return train_and_test(cfg, build_model, test_ckpt="best")


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = finetune(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
