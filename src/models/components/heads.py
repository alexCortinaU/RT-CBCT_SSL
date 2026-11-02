from typing import Optional, Sequence

import torch
from torch import nn


class MLPHead(nn.Module):
    """Three hidden layers (Linear -> BatchNorm -> ReLU) followed by a linear output layer.

    In the manuscript the hidden sizes are (256, 128, 64) and the input is the CLS token.
    """

    def __init__(self, input_size: int, hidden_layers: Sequence[int], output_size: int) -> None:
        super().__init__()
        lin1_size, lin2_size, lin3_size = hidden_layers
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))


def build_head(input_size: int, output_size: int, hidden_layers: Optional[Sequence[int]] = None) -> nn.Module:
    """MLP head if `hidden_layers` is given, otherwise a single linear layer (linear probe)."""
    if hidden_layers:
        return MLPHead(input_size, hidden_layers, output_size)
    return nn.Linear(input_size, output_size, bias=True)


class EncoderWithHead(nn.Module):
    """Wrap an image encoder and a head that is applied to the CLS token.

    Used as the base model that LoRA adapters are attached to.
    """

    def __init__(self, image_encoder: nn.Module, output_size: int = 2,
                 hidden_layers: Optional[Sequence[int]] = None):
        super().__init__()
        self.image_encoder = image_encoder
        self.head = build_head(image_encoder.embed_dim, output_size, hidden_layers)

    def forward(self, x):
        _, cls_tkn = self.image_encoder(x, return_cls_token=True)
        return self.head(cls_tkn)
