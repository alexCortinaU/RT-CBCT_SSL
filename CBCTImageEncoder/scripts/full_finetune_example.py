"""Minimal example: full-finetune a pretrained encoder + linear head for classification
(no LoRA -- every encoder weight is trainable). Same shape as lora_finetune_example.py,
without peft: a short, readable reference, not a general training framework.

    python scripts/full_finetune_example.py --data_dir /path/to/nifti --labels_csv labels.csv \
        --model_size small --checkpoints_dir checkpoints --model_out finetuned_model.pt

`labels_csv` needs two columns: `filename` (a .nii.gz under --data_dir) and `label`
(an integer class index).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import load_pretrained  # noqa: E402

# Reuse the dataset + model classes from the LoRA example -- identical data format and
# forward pass, the only difference is whether the encoder's weights are frozen.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lora_finetune_example import ClassifierModel, LabeledCBCTDataset  # noqa: E402


@dataclass
class Config:
    data_dir: str
    labels_csv: str
    checkpoints_dir: str = "checkpoints"
    model_size: str = "small"
    model_out: str = "finetuned_model.pt"
    num_classes: int = 2
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-5  # full finetuning needs a much smaller LR than the LoRA example's head-only 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: Config):
    encoder = load_pretrained(cfg.model_size, cfg.checkpoints_dir)
    model = ClassifierModel(encoder, cfg.num_classes).to(cfg.device)
    # No freezing here -- every parameter (encoder + head) is trainable.

    dataset = LabeledCBCTDataset(cfg.data_dir, cfg.labels_csv)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for volumes, labels in loader:
            volumes, labels = volumes.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            logits = model(volumes)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: loss={epoch_loss / len(loader):.4f}")

    # --- Saving ---: the whole model's weights (encoder + head), plain state_dict.
    torch.save(model.state_dict(), cfg.model_out)
    print(f"Saved finetuned model to {cfg.model_out}")


def load_finetuned(model_size: str, checkpoints_dir: str, model_path: str, num_classes: int = 2) -> nn.Module:
    """--- Loading it back ---: rebuild the same architecture, then load the state_dict.
    Note this loads the *finetuned* encoder weights too, not the original pretrained
    ones -- `load_pretrained` here just gives the right architecture to load into.
    """
    encoder = load_pretrained(model_size, checkpoints_dir)
    model = ClassifierModel(encoder, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--model_size", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--model_out", default="finetuned_model.pt")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(Config(**vars(args)))
