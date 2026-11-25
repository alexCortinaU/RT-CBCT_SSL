"""Evaluate a fully finetuned (or linear probing) checkpoint on the test set.

Example:
    python src/inference/finetune.py experiment=feeding_tube/full_finetune model_name=vit-s \
        checkpoint_path=/path/to/run/checkpoints/epoch_012.ckpt
"""
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import extras  # noqa: E402
from src.utils.checkpoints import load_finetuned_weights, load_pretrained_encoder  # noqa: E402
from src.utils.inference import run_inference  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference/finetune.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    # the pretrained encoder only provides the architecture; all weights come from the checkpoint
    image_encoder = load_pretrained_encoder(cfg.pretrained_ckpt)
    model = hydra.utils.instantiate(cfg.model, image_encoder=image_encoder)
    model = load_finetuned_weights(model, cfg.checkpoint_path)
    run_inference(cfg, model, metadata={'pretrained_ckpt': cfg.pretrained_ckpt,
                                        'checkpoint_path': cfg.checkpoint_path})


if __name__ == "__main__":
    main()
