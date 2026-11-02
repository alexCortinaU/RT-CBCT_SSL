"""RankMe of the CLS-token embeddings of a pretraining checkpoint (Garrido et al., 2023).

Embeds every CBCT listed in ``csv_file`` with the encoder of a SimMIM checkpoint, and saves the
singular values, RankMe and normalised RankMe (RankMe / embedding dimension). In the manuscript
this was computed for every pretraining epoch on the feeding tube validation set.

Example:
    python src/compute_rankme.py model_name=vit-s pretrained_epoch=99 csv_file=rankme_images.csv
"""
import json
from pathlib import Path
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
import torchio as tio
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.checkpoints import load_pretrained_encoder  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def compute_svd_and_rankme(latents: List[torch.Tensor],
                           center_z: bool = False) -> Tuple[np.ndarray, float, float]:
    """Singular values, RankMe and normalised RankMe of L2-normalised embeddings.

    :param latents: List of (B, D) embedding tensors.
    :param center_z: Subtract the mean embedding before normalisation.
    """
    z = torch.cat(latents, dim=0)
    if center_z:
        z = z - z.mean(dim=0, keepdim=True)
    z = torch.nn.functional.normalize(z, dim=1).cpu().numpy()
    singular_values = np.linalg.svd(z, full_matrices=False)[1]
    epsilon = 1e-10
    pk = singular_values / (np.linalg.norm(singular_values, ord=1) + epsilon)
    rankme = float(np.exp(-np.sum(pk * np.log(pk + epsilon))))
    return singular_values, rankme, rankme / z.shape[1]


class CBCTNumpyDataset(Dataset):
    """Preprocessed volumes listed in the ``cbct_path_npy`` column of a csv inside ``data_path``."""

    def __init__(self, data_path: str, csv_file: str, transforms=None):
        self.data_path = Path(data_path)
        csv_path = self.data_path / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV file {csv_path} does not exist')
        self.df = pd.read_csv(csv_path)
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = tio.ScalarImage(tensor=np.load(self.data_path / self.df['cbct_path_npy'].iloc[idx]))
        if self.transform:
            img = self.transform(img)
        return img.data


@hydra.main(version_base="1.3", config_path="../configs", config_name="compute_rankme.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    device = torch.device(f"cuda:{cfg.get('device', 0)}" if torch.cuda.is_available() else 'cpu')

    dataset = CBCTNumpyDataset(data_path=cfg.data_path, csv_file=cfg.csv_file)
    log.info(f"Dataset of length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                            pin_memory=True, shuffle=False, drop_last=False)

    image_encoder = load_pretrained_encoder(cfg.pretrained_ckpt).to(device).eval()

    cls_tokens = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Extracting image embeddings'):
            _, cls_tkn = image_encoder(images.to(device), return_cls_token=True)
            cls_tokens.append(cls_tkn.view(-1, image_encoder.embed_dim).cpu())

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_embeddings:
        np.save(output_dir / 'image_embeddings.npy', torch.cat(cls_tokens, dim=0).numpy())

    singular_values, rankme, normalized_rankme = compute_svd_and_rankme(cls_tokens, center_z=cfg.center_latents)
    log.info(f'RankMe: {rankme:.2f}, normalized RankMe: {normalized_rankme:.3f}')
    np.save(output_dir / 'singular_values.npy', singular_values)

    fig = plt.figure()
    plt.plot(np.log(singular_values))
    plt.xlabel('Singular value rank index')
    plt.ylabel('Log singular value')
    fig.savefig(output_dir / 'singular_values.png')
    plt.close(fig)

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({'pretrained_ckpt': cfg.pretrained_ckpt,
                   'pretrained_epoch': cfg.pretrained_epoch,
                   'csv_file': cfg.csv_file,
                   'center_latents': cfg.center_latents,
                   'rankme': rankme,
                   'normalized_rankme': normalized_rankme}, f, indent=4)
    log.info(f'Saved RankMe results to {output_dir}')


if __name__ == "__main__":
    main()
