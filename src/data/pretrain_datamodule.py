"""Unlabelled CBCT data for SimMIM pretraining.

Expected layout under ``data_path``::

    <csv_file>            one row per CBCT with at least the columns
                            split            "train" or "val"
                            cbct_path_npy    preprocessed volume (.npy, shape (1, H, W, D)), used when
                                             read_numpy=True (as in the manuscript)
                          and, for on-the-fly preprocessing (read_numpy=False):
                            cbct_path        raw CBCT image (any format readable by torchio)
                            body_seg_path    body mask used to sample the crop inside the patient
                            margins_to_crop  str of a 6-tuple of crop margins around the body
    <files referenced by the csv, relative to data_path>

Preprocessed volumes are expected to be resampled to ``voxel_size``, clamped to [-1000, 1000] HU,
rescaled to [-1, 1] and cropped to ``crop_size``. With ``read_numpy=False`` this is done on the fly
by :class:`PatchBasedTransform`.
"""
import ast
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torchio as tio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.masking import SimMIMTransform, collate_simmim


class _ToleranceSubject(tio.Subject):
    """Subject that tolerates small affine/origin differences between the CBCT and its body mask."""

    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-1
        kwargs['absolute_tolerance'] = 1e-1
        return super().check_consistent_attribute(*args, **kwargs)


def read_exclude_file(exclude_file: Optional[str]) -> List[str]:
    """Read a text file with one ``cbct_path`` per line (images to leave out)."""
    if exclude_file is None:
        return []
    with open(exclude_file) as f:
        return [line.strip() for line in f if line.strip()]


class CBCTPretrainDataset(Dataset):
    """CBCT volumes for self-supervised pretraining.

    :param data_path: Root folder of the dataset (see module docstring).
    :param csv_file: Name of the csv file inside ``data_path``.
    :param train: Use the ``train`` split, otherwise ``val``.
    :param transform: Transform applied to each sample (e.g. :class:`SimMIMTransform`).
    :param sample: Optional fraction of the training set to use.
    :param examples: Optional indices of validation samples (used to plot reconstructions).
    :param read_numpy: Read preprocessed ``.npy`` volumes instead of raw images.
    :param exclude_file: Optional text file listing ``cbct_path`` values to exclude.
    """

    def __init__(self,
                 data_path: str,
                 csv_file: str = 'pretrain_cbcts.csv',
                 train: bool = True,
                 transform=None,
                 sample: Optional[float] = None,
                 examples: Optional[Sequence[int]] = None,
                 read_numpy: bool = False,
                 exclude_file: Optional[str] = None):
        self.data_path = Path(data_path)
        csv_path = self.data_path / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f'File {csv_path} does not exist')

        self.transform = transform
        self.read_numpy = read_numpy
        self.split = 'train' if train else 'val'
        self.df = pd.read_csv(csv_path)

        if examples is not None:
            df = self.df[self.df.split == 'val'].reset_index(drop=True)
            self.df = df.iloc[list(examples)].reset_index(drop=True)
        else:
            self.df = self.df[self.df.split == self.split].reset_index(drop=True)
            excluded = read_exclude_file(exclude_file)
            if excluded:
                n_before = len(self.df)
                self.df = self.df[~self.df.cbct_path.isin(excluded)].reset_index(drop=True)
                print(f'Excluded {n_before - len(self.df)} images listed in {exclude_file}')
            if train and sample is not None:
                self.df = self.df.sample(frac=sample, random_state=42).reset_index(drop=True)

        if not self.read_numpy:
            self.df['margins_to_crop'] = self.df['margins_to_crop'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.read_numpy:
            img_nda = np.load(self.data_path / self.df['cbct_path_npy'].iloc[idx])
            subject = tio.ScalarImage(tensor=img_nda)
        else:
            img_path = self.data_path / self.df['cbct_path'].iloc[idx]
            label_path = self.data_path / self.df['body_seg_path'].iloc[idx]
            subject = _ToleranceSubject(image=tio.ScalarImage(img_path),
                                        label=tio.LabelMap(label_path))
            # resample the body mask to the image grid, then crop around the body
            subject = tio.Resample(target='image', exclude=['image'])(subject)
            subject = tio.Crop(self.df['margins_to_crop'].iloc[idx])(subject)

        if self.transform:
            subject = self.transform(subject)

        if isinstance(subject, tuple):  # (image, SimMIM mask)
            return subject[0].data, subject[1]
        return subject


class PatchBasedTransform:
    """On-the-fly preprocessing: intensity/spacing transforms, padding and a random crop of
    ``crop_size`` centred inside the body mask."""

    def __init__(self, crop_size=(256, 224, 64), transforms=None):
        self.crop_size = crop_size
        self.transforms = transforms
        self.sampler = tio.data.LabelSampler(patch_size=crop_size, label_name='label')

    def __call__(self, subject, **kwds):
        subject = self.transforms(subject)
        # ensure the image is at least crop_size for patch extraction
        pad = [max(0, self.crop_size[i] - subject.image.shape[i + 1] + 1) for i in range(3)]
        pad_transform = tio.Compose([
            tio.transforms.Pad(pad, padding_mode=-1, exclude=['label']),
            tio.transforms.Pad(pad, padding_mode=0, exclude=['image'])
        ])
        subject = pad_transform(subject)
        patch = next(iter(self.sampler(subject, num_patches=1)))
        return patch.image.data


class SimMIMPretrainDataModule(LightningDataModule):
    """Datamodule returning ``(image, mask)`` batches for SimMIM.

    :param data_path: Root folder of the dataset.
    :param csv_file: Name of the csv file inside ``data_path``.
    :param batch_size: Global batch size (divided by the number of devices).
    :param voxel_size: Target voxel spacing (on-the-fly preprocessing only).
    :param crop_size: Input volume size.
    :param patch_size: ViT patch size.
    :param mask_ratio: Fraction of mask patches that are masked.
    :param mask_patch_size: Side length of the masked blocks (voxels).
    :param transforms: Optional torchio transforms replacing the default on-the-fly preprocessing.
    :param sample: Optional fraction of the training set to use.
    :param examples: Indices of validation samples to plot reconstructions for.
    :param read_numpy: Read preprocessed ``.npy`` volumes (recommended).
    :param exclude_file: Optional text file listing ``cbct_path`` values to exclude.
    """

    def __init__(
        self,
        data_path: str,
        csv_file: str = 'pretrain_cbcts.csv',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        voxel_size: Tuple[float, float, float] = (1, 1, 2.5),
        crop_size: Tuple[int, int, int] = (256, 224, 64),
        patch_size: int = 16,
        mask_ratio: float = 0.6,
        mask_patch_size: int = 32,
        transforms=None,
        sample: Optional[float] = None,
        examples: Optional[Sequence[int]] = (0, 1, 2, 3),
        read_numpy: bool = False,
        exclude_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if transforms is None:
            transforms = tio.Compose([
                tio.transforms.ToCanonical(),
                tio.transforms.Resample(voxel_size),
                tio.transforms.Clamp(-1000, 1000),
                tio.transforms.RescaleIntensity(out_min_max=(-1.0, 1.0), in_min_max=(-1000, 1000))
            ], exclude=['label'])
        patch_transform = PatchBasedTransform(crop_size=crop_size, transforms=transforms)

        simmim_transform = SimMIMTransform(img_size=crop_size,
                                           mask_ratio=mask_ratio,
                                           mask_patch_size=mask_patch_size,
                                           model_patch_size=patch_size,
                                           transforms=None if read_numpy else patch_transform)

        dataset_kwargs = dict(data_path=data_path, csv_file=csv_file, read_numpy=read_numpy,
                              transform=simmim_transform)
        self.data_train = CBCTPretrainDataset(train=True, sample=sample, exclude_file=exclude_file,
                                              **dataset_kwargs)
        self.data_val = CBCTPretrainDataset(train=False, exclude_file=exclude_file, **dataset_kwargs)
        self.examples = (CBCTPretrainDataset(train=False, examples=examples, **dataset_kwargs)
                         if examples else None)

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _dataloader(self, dataset, shuffle: bool, drop_last: bool, batch_size: Optional[int] = None):
        return DataLoader(
            dataset=dataset,
            collate_fn=collate_simmim,
            batch_size=batch_size or self.batch_size_per_device,
            drop_last=drop_last,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_train, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_val, shuffle=False, drop_last=True)

    def examples_dataloader(self) -> DataLoader[Any]:
        """A single batch with the example volumes used to plot reconstructions."""
        return self._dataloader(self.examples, shuffle=False, drop_last=False, batch_size=len(self.examples))
