"""Train a randomly initialised ViT and head on a downstream task (no pretraining baseline).

Example:
    python src/train_supervised.py experiment=feeding_tube/supervised model.model_name=vit_small
"""
from typing import Optional

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import extras, get_metric_value, task_wrapper  # noqa: E402
from src.utils.training import class_weights_kwargs, train_and_test  # noqa: E402


def build_model(cfg: DictConfig, datamodule: LightningDataModule) -> LightningModule:
    return hydra.utils.instantiate(cfg.model, **class_weights_kwargs(datamodule))


@task_wrapper
def train_supervised(cfg: DictConfig):
    # the best checkpoint (see callbacks.model_checkpoint) is evaluated on the test set
    return train_and_test(cfg, build_model, test_ckpt="best")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_supervised.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train_supervised(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
