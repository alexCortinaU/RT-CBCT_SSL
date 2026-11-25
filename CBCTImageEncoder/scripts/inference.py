"""Get CLS-token embeddings for a folder of CBCT NIfTI volumes.

    python scripts/inference.py --data_dir /path/to/nifti/scans --output_dir /path/to/out \
        --model_size small --checkpoints_dir checkpoints

For each `<case>.nii.gz` found (recursively) under --data_dir, saves
`<output_dir>/<case>.npy` -- the CLS embedding, shape (embed_dim,).

Preprocessing (this file's `NiftiCBCTDataset`) is a **simple centered crop**: reorient,
resample to the model's voxel spacing, then crop/pad to its expected volume size, centered
on the volume -- no body-segmentation-aware cropping. This is fine for scans where the
anatomy of interest is already roughly centered in the field of view. If your scans have
a lot of surrounding table/immobilization mask/background, or the anatomy of interest
isn't centered, a plain center crop may cut it off or leave it off-center after resizing.
**If that matters for your data, preprocess offline first** (crop around the body/anatomy
of interest yourself, with whatever segmentation approach fits your data) and point
--data_dir at the already-cropped volumes -- this script's center crop will then be a
near-no-op.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import load_pretrained  # noqa: E402
from src.preprocessing import DEFAULT_IMG_SIZE, DEFAULT_VOXEL_SIZE, build_transform  # noqa: E402


class NiftiCBCTDataset(Dataset):
    """Reads every `*.nii.gz` under `data_dir`, preprocesses it "online" (per-item, in
    `__getitem__` -- plays nicely with DataLoader(num_workers>0)) to the model's expected
    volume size and voxel spacing. See module docstring re: this being a *simple centered*
    crop, and `src/preprocessing.py` for the exact transform pipeline.
    """

    def __init__(self, data_dir: str, img_size=DEFAULT_IMG_SIZE, voxel_size=DEFAULT_VOXEL_SIZE):
        self.paths = sorted(Path(data_dir).rglob("*.nii.gz"))
        if not self.paths:
            raise FileNotFoundError(f"No .nii.gz files found under {data_dir}")
        self.transform = build_transform(img_size, voxel_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.transform(tio.ScalarImage(str(path)))
        case_id = path.name[: -len(".nii.gz")]
        return image.data, case_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Folder of CBCT NIfTI (.nii.gz) volumes.")
    parser.add_argument("--output_dir", required=True, help="Where to save <case>.npy CLS embeddings.")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--model_size", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_pretrained(args.model_size, args.checkpoints_dir).to(args.device)

    dataset = NiftiCBCTDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print(f"Found {len(dataset)} volume(s) under {args.data_dir}. Using device={args.device}.")
    n_done = 0
    with torch.no_grad():
        for volumes, case_ids in loader:
            volumes = volumes.to(args.device)
            _, cls_embeddings = model(volumes, return_cls_token=True)
            cls_embeddings = cls_embeddings.cpu().numpy()
            for case_id, embedding in zip(case_ids, cls_embeddings):
                np.save(output_dir / f"{case_id}.npy", embedding)
                n_done += 1

    print(f"Done. Saved {n_done} embedding(s) (shape ({model.embed_dim},) each) to {output_dir}")


if __name__ == "__main__":
    main()
