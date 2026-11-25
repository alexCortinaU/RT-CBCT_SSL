"""Minimal example: LoRA-finetune a pretrained encoder + linear head for classification.

Not a general training framework -- a short, readable reference showing the three parts
that matter: wrapping the frozen encoder in a LoRA adapter (peft), a plain training loop,
and how to save/reload the trained adapter afterward.

    python scripts/lora_finetune_example.py --data_dir /path/to/nifti --labels_csv labels.csv \
        --model_size small --checkpoints_dir checkpoints --adapter_out lora_adapter/

`labels_csv` needs two columns: `filename` (a .nii.gz under --data_dir) and `label`
(an integer class index).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchio as tio
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import load_pretrained  # noqa: E402
from src.preprocessing import build_transform  # noqa: E402


@dataclass
class Config:
    data_dir: str
    labels_csv: str
    checkpoints_dir: str = "checkpoints"
    model_size: str = "small"
    adapter_out: str = "lora_adapter"
    num_classes: int = 2
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LabeledCBCTDataset(Dataset):
    """Pairs each row of `labels_csv` (columns: filename, label) with the preprocessed
    volume at `<data_dir>/<filename>`. Same "simple centered crop" preprocessing as
    scripts/inference.py -- see that file's docstring."""

    def __init__(self, data_dir: str, labels_csv: str):
        self.data_dir = Path(data_dir)
        self.rows = pd.read_csv(labels_csv)
        self.transform = build_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        image = self.transform(tio.ScalarImage(str(self.data_dir / row["filename"])))
        return image.data, int(row["label"])


class ClassifierModel(nn.Module):
    """Frozen encoder (or LoRA-wrapped, see below) -> CLS token -> linear head."""

    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x):
        _, cls_token = self.encoder(x, return_cls_token=True)
        return self.head(cls_token)


def train(cfg: Config):
    encoder = load_pretrained(cfg.model_size, cfg.checkpoints_dir)
    model = ClassifierModel(encoder, cfg.num_classes)

    # Wrap in a LoRA adapter: encoder's linear layers get small trainable low-rank
    # matrices, everything else (including the head, via modules_to_save) is trained
    # normally -- the base encoder's own weights never change.
    lora_config = LoraConfig(target_modules="all-linear", modules_to_save=["head"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(cfg.device)

    dataset = LabeledCBCTDataset(cfg.data_dir, cfg.labels_csv)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
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

    # --- Saving the adapter ---
    # Only the small LoRA delta weights + head are saved (a few MB), not the base encoder.
    model.save_pretrained(cfg.adapter_out)
    print(f"Saved LoRA adapter to {cfg.adapter_out}")


def load_finetuned(model_size: str, checkpoints_dir: str, adapter_dir: str, num_classes: int = 2) -> nn.Module:
    """--- Loading the adapter back (e.g. in a fresh process/script) ---
    Rebuild the exact same frozen base model, then attach the saved adapter to it.
    """
    encoder = load_pretrained(model_size, checkpoints_dir)
    base_model = ClassifierModel(encoder, num_classes)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--model_size", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--adapter_out", default="lora_adapter")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(Config(**vars(args)))
