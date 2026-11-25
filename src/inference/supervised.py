"""Evaluate a supervised (no pretraining) checkpoint on the test set.

Example:
    python src/inference/supervised.py experiment=feeding_tube/supervised \
        checkpoint_path=/path/to/run/checkpoints/epoch_020.ckpt
"""
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import extras  # noqa: E402
from src.utils.checkpoints import load_finetuned_weights  # noqa: E402
from src.utils.inference import run_inference  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference/supervised.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    model = hydra.utils.instantiate(cfg.model)
    model = load_finetuned_weights(model, cfg.checkpoint_path)
    run_inference(cfg, model, metadata={'checkpoint_path': cfg.checkpoint_path})


if __name__ == "__main__":
    main()
