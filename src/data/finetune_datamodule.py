"""Labelled CBCT data for the downstream tasks (feeding tube and clinical characteristics).

Expected layout under ``data_path``: a training csv (train/val rows), a test csv, and the
images they reference (paths relative to ``data_path``). Columns used:

    split_finetune    "train" or "val" (the test csv rows must use "val")
    cbct_path_npy     preprocessed volume (.npy), used when read_numpy=True (as in the manuscript)
    cbct_path, margins_to_crop
                      raw image and body crop margins, used when read_numpy=False
    patient_id        patient identifier; training subsets are sampled per patient
    split_finetune_ablation
                      (optional) nested subset index in {16, 8, 4, 2, 1}, used when
                      ``train_subsplit`` is one of these integers
    <label column>    ``has_ft`` for the feeding tube task, or the column in ``TASKS`` for the
                      clinical tasks
"""
import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchio as tio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# downstream task name -> label column in the csv
TASKS = {
    'feeding_tube_classification': 'has_ft',
    'age_regression': 'age',
    'sex_classification': 'sex',
    't_stage_classification': 'cTstage',
    'n_stage_classification': 'cNstage',
    'tumorsite_classification': 'tumorsite',
}

# label encodings (classification) and normalisation (age regression)
TASKS_ENCODING = {
    'feeding_tube_classification': None,  # already encoded as 0/1
    'age_regression': {'train_max': 92},  # ages are divided by the max age in the training set
    'sex_classification': {'male': 0, 'female': 1},
    't_stage_classification': {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
    'n_stage_classification': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
    'tumorsite_classification': {'c. oris': 0, 'hypopharynx': 1, 'larynx': 2, 'oropharynx': 3, 'other': 4},
}

ABLATION_SUBSPLITS = [16, 8, 4, 2, 1]


def num_classes(task: str) -> int:
    """Number of classes of a classification task (2 for the feeding tube)."""
    encoding = TASKS_ENCODING[task]
    return 2 if encoding is None else len(encoding)


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """One loss weight per class index: n_samples / (n_present_classes * n_samples_of_class),
    as sklearn's "balanced" weights, and 0 for classes absent from `labels`."""
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    present = counts > 0
    weights = np.zeros(n_classes)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float)


def subsample_train_set(df: pd.DataFrame, train_subsplit: Union[int, float], seed: int) -> pd.DataFrame:
    """Select the training subset used for the annotation-budget ablation.

    - ``train_subsplit`` in {16, 8, 4, 2, 1} (int): keep the rows whose
      ``split_finetune_ablation`` is >= train_subsplit (nested, predefined subsets);
      ``1`` keeps the whole set.
    - ``0 < train_subsplit <= 1`` (float): randomly sample this fraction of *patients* with
      ``seed`` (all CBCTs of a sampled patient are kept).
    """
    if isinstance(train_subsplit, int) and train_subsplit in ABLATION_SUBSPLITS:
        if train_subsplit == 1 and 'split_finetune_ablation' not in df.columns:
            return df
        subsplits = [s for s in ABLATION_SUBSPLITS if s >= train_subsplit]
        return df[df.split_finetune_ablation.isin(subsplits)].reset_index(drop=True)
    if isinstance(train_subsplit, float) and 0.0 < train_subsplit <= 1.0:
        patients = pd.Series(df.patient_id.unique())
        sampled = patients.sample(frac=train_subsplit, random_state=seed).unique()
        return df[df.patient_id.isin(sampled)].reset_index(drop=True)
    raise ValueError(f'Invalid train_subsplit value: {train_subsplit}')


class CBCTLabelledDataset(Dataset):
    """CBCT volumes with a downstream label.

    :param data_path: Root folder of the dataset.
    :param csv_file: Csv file (inside ``data_path``) to read.
    :param task: Key of ``TASKS``.
    :param train: Use the ``train`` rows (``split_finetune``), otherwise ``val``.
    :param train_subsplit: Training subset, see :func:`subsample_train_set` (training set only).
    :param seed: Seed for the patient sampling; a random one is drawn if None.
    :param transform: Transform applied to each image.
    :param read_numpy: Read preprocessed ``.npy`` volumes instead of raw images.
    :param return_paths: Additionally return the image path of each sample.
    """

    def __init__(self,
                 data_path: str,
                 csv_file: str,
                 task: str,
                 train: bool = True,
                 train_subsplit: Union[int, float] = 1,
                 seed: Optional[int] = None,
                 transform=None,
                 read_numpy: bool = False,
                 return_paths: bool = False):
        self.data_path = Path(data_path)
        csv_path = self.data_path / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f'File {csv_path} does not exist')
        if task not in TASKS:
            raise ValueError(f'Unknown task {task}, expected one of {list(TASKS)}')

        self.task = task
        self.label = TASKS[task]
        self.is_classification = 'classification' in task
        self.transform = transform
        self.read_numpy = read_numpy
        self.return_paths = return_paths
        self.split = 'train' if train else 'val'
        self.seed = datetime.now().microsecond if seed is None else seed
        self.train_subsplit = train_subsplit

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.split_finetune == self.split].reset_index(drop=True)
        if train_subsplit is not None and train:
            self.df = subsample_train_set(self.df, train_subsplit, self.seed)

        # encode / normalise labels and drop rows without one
        self.df = self.df.dropna(subset=[self.label]).reset_index(drop=True)
        encoding = TASKS_ENCODING[task]
        if task == 'age_regression':
            self.df[self.label] = self.df[self.label] / encoding['train_max']
        elif encoding is not None:
            self.df[self.label] = self.df[self.label].map(encoding)

        if not self.read_numpy:
            self.df['margins_to_crop'] = self.df['margins_to_crop'].apply(ast.literal_eval)

        self.class_weights = None
        self.samples_weight = None
        if self.is_classification:
            self.class_weights = balanced_class_weights(self.df[self.label].values, num_classes(task))
            missing = (self.class_weights == 0).nonzero().flatten().tolist()
            if missing and train:
                log.warning(f'{task}: classes {missing} are absent from the training set; their loss weight is 0')
            # per-sample weights for an optional WeightedRandomSampler
            inverse_freq = 1. / self.df[self.label].value_counts()
            self.samples_weight = torch.from_numpy(self.df[self.label].map(inverse_freq).values).double()
            counts = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in self.df[self.label].value_counts().items())
            print(f'{csv_file} [{self.split}] (n={len(self.df)}): {counts}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.read_numpy:
            img = tio.ScalarImage(tensor=np.load(self.data_path / self.df['cbct_path_npy'].iloc[idx]))
        else:
            img = tio.ScalarImage(self.data_path / self.df['cbct_path'].iloc[idx])
            img = tio.Crop(self.df['margins_to_crop'].iloc[idx])(img)

        if self.transform:
            img = self.transform(img)

        target = self.df[self.label].iloc[idx]
        if self.is_classification:
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = torch.Tensor([target])

        if self.return_paths:
            path_col = 'cbct_path_npy' if self.read_numpy else 'cbct_path'
            return img.data, target, self.df[path_col].iloc[idx]
        return img.data, target


class FinetuneDataModule(LightningDataModule):
    """Train / val / test dataloaders for a downstream task.

    :param data_path: Root folder of the dataset.
    :param task: Key of ``TASKS``.
    :param train_csv: Csv with the ``train`` and ``val`` rows.
    :param test_csv: Csv with the test rows.
    :param linear_probe_csv: If set, replaces ``train_csv`` (no subsampling). In the manuscript this
        is the feeding tube *validation* set, re-split for the per-epoch linear probing.
    :param batch_size: Global batch size (divided by the number of devices).
    :param train_subsplit: Training subset for the annotation-budget ablation.
    :param seed: Seed for the training subset sampling.
    :param train_weighted_sampler: Oversample minority classes instead of weighting the loss.
    :param online_preprocessing: Resample/crop/normalise raw images on the fly (``read_numpy=False``).
    :param augmentations: Optional torchio transforms applied to the training images.
    :param read_numpy: Read preprocessed ``.npy`` volumes (as in the manuscript).
    :param return_test_paths: Return the image paths with the test samples (for inference).
    """

    def __init__(
            self,
            data_path: str,
            task: str,
            train_csv: str,
            test_csv: str,
            linear_probe_csv: Optional[str] = None,
            batch_size: int = 10,
            num_workers: int = 0,
            pin_memory: bool = False,
            voxel_size: Tuple[float, float, float] = (1, 1, 2.5),
            crop_size: Tuple[int, int, int] = (256, 224, 64),
            train_subsplit: Union[int, float] = 1,
            seed: Optional[int] = None,
            train_weighted_sampler: bool = False,
            online_preprocessing: bool = False,
            augmentations: Optional[tio.Transform] = None,
            read_numpy: bool = False,
            return_test_paths: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if online_preprocessing:
            self.init_transforms = tio.Compose([
                tio.transforms.ToCanonical(),
                tio.transforms.Resample(voxel_size),
                tio.transforms.CropOrPad(crop_size, padding_mode=-1000),
                tio.transforms.Clamp(-1000, 1000),
                tio.transforms.RescaleIntensity(out_min_max=(-1.0, 1.0), in_min_max=(-1000, 1000)),
            ])
        else:
            self.init_transforms = tio.Compose([])
        train_transforms = (tio.Compose([self.init_transforms, augmentations])
                            if augmentations else self.init_transforms)

        common = dict(data_path=data_path, task=task, read_numpy=read_numpy)
        if linear_probe_csv is not None:
            train_csv, train_subsplit = linear_probe_csv, None
        self.data_train = CBCTLabelledDataset(csv_file=train_csv, train=True, train_subsplit=train_subsplit,
                                              seed=seed, transform=train_transforms, **common)
        self.data_val = CBCTLabelledDataset(csv_file=train_csv, train=False, transform=self.init_transforms,
                                            **common)
        self.data_test = CBCTLabelledDataset(csv_file=test_csv, train=False, transform=self.init_transforms,
                                             return_paths=return_test_paths, **common)
        self.class_weights = self.data_train.class_weights
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _dataloader(self, dataset, **kwargs) -> DataLoader[Any]:
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          **kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.hparams.train_weighted_sampler:
            sampler = WeightedRandomSampler(self.data_train.samples_weight, len(self.data_train), replacement=True)
            return self._dataloader(self.data_train, drop_last=True, sampler=sampler)
        return self._dataloader(self.data_train, drop_last=True, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_val, drop_last=False, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_test, drop_last=False, shuffle=False)
