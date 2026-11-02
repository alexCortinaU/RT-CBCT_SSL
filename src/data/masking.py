# --------------------------------------------------------
# Adapted from https://github.com/microsoft/SimMIM :
#   SimMIM
#   Copyright (c) 2021 Microsoft
#   Licensed under The MIT License [see LICENSE for details]
#   Written by Zhenda Xie
# --------------------------------------------------------
"""Random block masking for SimMIM, extended to 3D volumes."""
import numpy as np
from torch.utils.data._utils.collate import default_collate


class MaskGenerator:
    """Mask a random `mask_ratio` of the (mask_patch_size)^3 blocks of a volume.

    The returned mask has one entry per ViT token (img_size / model_patch_size per axis),
    with 1 marking a masked token.
    """

    def __init__(self, img_size=(256, 224, 64), mask_patch_size=32, model_patch_size=16, mask_ratio=0.6):
        for dim in img_size:
            assert dim % mask_patch_size == 0, f'{dim} % {mask_patch_size} = {dim % mask_patch_size}'
        assert mask_patch_size % model_patch_size == 0

        self.rand_size = tuple(dim // mask_patch_size for dim in img_size)
        self.scale = mask_patch_size // model_patch_size
        self.token_count = int(np.prod(self.rand_size))
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape(self.rand_size)
        return mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)


class SimMIMTransform:
    """Apply optional image transforms and return ``(image, mask)``."""

    def __init__(self,
                 img_size=(256, 224, 64),
                 mask_ratio=0.6,
                 mask_patch_size=32,
                 model_patch_size=16,
                 transforms=None):
        self.transform = transforms
        self.mask_generator = MaskGenerator(
            img_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

    def __call__(self, img):
        if self.transform:
            img = self.transform(img)
        return img, self.mask_generator()


def collate_simmim(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    ret = []
    for item_idx in range(len(batch[0][0])):
        if batch[0][0][item_idx] is None:
            ret.append(None)
        else:
            ret.append(default_collate([sample[0][item_idx] for sample in batch]))
    return ret
