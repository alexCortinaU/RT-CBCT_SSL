"""Training loop shared by the downstream scripts (finetune.py, finetune_lora.py, train_supervised.py)."""
from typing import Any, Callable, Dict, List, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def train_and_test(cfg: DictConfig,
                   build_model: Callable[[DictConfig, LightningDataModule], LightningModule],
                   test_ckpt: str = "best") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit a downstream model and evaluate it on the test set.

    :param cfg: A DictConfig configuration composed by Hydra.
    :param build_model: Function returning the LightningModule given the config and datamodule.
    :param test_ckpt: ``"best"`` to test the best checkpoint of the ``ModelCheckpoint`` callback,
        ``"last"`` to test the weights at the end of training.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    log.info(f"Training dataset of length: {len(datamodule.data_train)}")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = build_model(cfg, datamodule)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = dict(trainer.callback_metrics)

    if cfg.get("test"):
        ckpt_path = None
        if test_ckpt == "best":
            ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        log.info(f"Starting testing! (weights: {ckpt_path or 'end of training'})")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    metric_dict = {**train_metrics, **trainer.callback_metrics}
    return metric_dict, object_dict


def class_weights_kwargs(datamodule: LightningDataModule) -> Dict[str, Any]:
    """Loss class weights for classification tasks (unless a weighted sampler is used)."""
    if datamodule.class_weights is None or datamodule.hparams.train_weighted_sampler:
        return {}
    return {"class_weights": datamodule.class_weights}
