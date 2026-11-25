"""Finetune a pretrained encoder with low-rank adaptation (LoRA) and an MLP head on the CLS token.

Example:
    python src/finetune_lora.py experiment=feeding_tube/lora model_name=vit-s
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
    # _convert_="partial" passes `lora_config.target_modules` as a list, as expected by peft
    return hydra.utils.instantiate(cfg.model, image_encoder=image_encoder, _convert_="partial",
                                   **class_weights_kwargs(datamodule))


@task_wrapper
def finetune_lora(cfg: DictConfig):
    # the best checkpoint (see callbacks.model_checkpoint) is evaluated on the test set; its LoRA
    # adapters are also saved in `lora_checkpoints/lora_best` by the LoraCheckpoint callback
    return train_and_test(cfg, build_model, test_ckpt="best")


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune_lora.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = finetune_lora(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
