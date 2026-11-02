"""Loading pretrained encoders and finetuned weights from Lightning checkpoints."""
import inspect
from typing import Dict

import torch
from lightning import LightningModule

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the ``_orig_mod.`` infix that ``torch.compile`` adds to parameter names."""
    return {k.replace('._orig_mod.', '.'): v for k, v in state_dict.items()}


def load_pretrained_encoder(ckpt_path: str) -> torch.nn.Module:
    """Return the ViT encoder of a SimMIM pretraining checkpoint."""
    from src.models.simmim_module import SimMIMModule

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # checkpoints from earlier versions of the code store a few extra (unused) hyperparameters
    accepted = inspect.signature(SimMIMModule.__init__).parameters
    hparams = {k: v for k, v in ckpt['hyper_parameters'].items() if k in accepted}
    hparams['compile'] = False

    model = SimMIMModule(**hparams)
    model.load_state_dict(strip_compile_prefix(ckpt['state_dict']))
    log.info(f'Loaded pretrained {hparams["model_name"]} encoder from {ckpt_path} (epoch {ckpt.get("epoch")})')
    return model.encoder


def load_finetuned_weights(model: LightningModule, ckpt_path: str) -> LightningModule:
    """Load the weights of a finetuned (or supervised) checkpoint into an instantiated model.

    The loss buffers (class weights) are skipped since they are not needed for inference.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k: v for k, v in strip_compile_prefix(ckpt['state_dict']).items()
                  if not k.startswith('loss.')}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith('loss.')]
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint {ckpt_path} does not match the model: '
                           f'missing keys {missing}, unexpected keys {unexpected}')
    log.info(f'Loaded finetuned weights from {ckpt_path} (epoch {ckpt.get("epoch")})')
    return model
