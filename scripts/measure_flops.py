"""FLOPs of one training step (forward + backward of one mini-batch) of the ViT encoder.

As for the manuscript, the step is measured on the encoder forward pass (``SimMIMModule.forward``,
average-pooled tokens) with a sum as loss; the 1x1x1 convolutional decoder is negligible.
The manuscript reports pretraining compute as PF-days:
    PF-days = FLOPs per step * steps per epoch * epochs / (1e15 * 86400)

Example:
    python scripts/measure_flops.py --model vit_small --batch-size 30
"""
import argparse

import rootutils
import torch
from lightning.pytorch.utilities import measure_flops

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.simmim_module import SimMIMModule  # noqa: E402


def measure_simmim_flops(model_name: str, batch_size: int, crop_size=(256, 224, 64), patch_size=16):
    with torch.device('meta'):
        model = SimMIMModule(model_name=model_name, optimizer=None, scheduler=None, num_epochs=1,
                             compile=False, cls_token=True, normalize_output=True, pos_embed_type='sincos',
                             crop_size=crop_size, patch_size=patch_size)
        x = torch.randn(batch_size, 1, *crop_size)

    model_fwd = lambda: model(x)  # noqa: E731
    fwd_flops = measure_flops(model, model_fwd)
    fwd_and_bwd_flops = measure_flops(model, model_fwd, lambda y: y.sum())
    return fwd_flops, fwd_and_bwd_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='vit_base', choices=['vit_tiny', 'vit_small', 'vit_base'])
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--crop-size', type=int, nargs=3, default=[256, 224, 64])
    args = parser.parse_args()

    fwd, fwd_bwd = measure_simmim_flops(args.model, args.batch_size, tuple(args.crop_size))
    print(f"{args.model}, batch size {args.batch_size}")
    print(f"FLOPs per step (forward only): {fwd:.3e}")
    print(f"FLOPs per step (forward and backward): {fwd_bwd:.3e}")
