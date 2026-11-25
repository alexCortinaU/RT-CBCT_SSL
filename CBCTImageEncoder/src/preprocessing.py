"""The preprocessing transform the model expects: reorient, resample to the training
voxel spacing, then a simple centered crop/pad to the training volume size, clamp, and
rescale to [-1, 1]. No body-segmentation-aware cropping -- see scripts/inference.py's
module docstring for what that means for your data.
"""

import torchio as tio

DEFAULT_IMG_SIZE = (256, 224, 64)
DEFAULT_VOXEL_SIZE = (1.0, 1.0, 2.5)


def build_transform(img_size=DEFAULT_IMG_SIZE, voxel_size=DEFAULT_VOXEL_SIZE) -> tio.Compose:
    return tio.Compose([
        tio.ToCanonical(),
        tio.Resample(voxel_size),
        tio.CropOrPad(img_size, padding_mode=-1000),  # centered by default (no mask given)
        tio.Clamp(-1000, 1000),
        tio.RescaleIntensity(out_min_max=(-1.0, 1.0), in_min_max=(-1000, 1000)),
    ])
