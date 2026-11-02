"""Evaluate LoRA adapters on the test set.

Example:
    python src/inference/lora.py experiment=feeding_tube/lora model_name=vit-s \
        lora_checkpoint_path=/path/to/run/lora_checkpoints
``lora_checkpoint_path`` can point to one adapter folder or to the ``lora_checkpoints`` folder of a
run, in which case the adapters of the best epoch (``lora_best``) are used.
"""
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig
from peft import PeftModel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.heads import EncoderWithHead  # noqa: E402
from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.checkpoints import load_pretrained_encoder  # noqa: E402
from src.utils.inference import run_inference  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def resolve_lora_checkpoint(path: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'LoRA checkpoint path {path} does not exist')
    if path.name == 'lora_checkpoints':
        if (path / 'lora_best').exists():
            return path / 'lora_best'
        # runs from earlier versions of the code saved lora_epoch_<N> / lora_final_<N>:
        # take the final one, else the latest epoch
        final = sorted(path.glob('lora_final_*'))
        path = final[-1] if final else max(path.glob('lora_epoch_*'), key=lambda p: int(p.name.split('_')[-1]))
    return path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference/lora.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    image_encoder = load_pretrained_encoder(cfg.pretrained_ckpt)
    output_size = cfg.model.num_classes if cfg.model.get('task') else 1
    model = EncoderWithHead(image_encoder=image_encoder, output_size=output_size,
                            hidden_layers=cfg.model.hidden_layers)
    lora_path = resolve_lora_checkpoint(cfg.lora_checkpoint_path)
    log.info(f'Loading LoRA weights from {lora_path}')
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    run_inference(cfg, model, metadata={'pretrained_ckpt': cfg.pretrained_ckpt,
                                        'lora_checkpoint_path': str(lora_path)})


if __name__ == "__main__":
    main()
