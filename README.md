<div align="center">

# Self-supervised pretraining for radiotherapy CBCT

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

Code accompanying *"Impact of Pretraining Length, Model Size, and Annotation Budget on Downstream
Clinical Tasks of Foundation Models in Radiotherapy"*.

3D vision transformers (ViT-t, ViT-s, ViT-b) are pretrained with masked image modeling
([SimMIM](https://arxiv.org/abs/2111.09886)) on cone-beam CT (CBCT) scans acquired during
radiotherapy, and finetuned with low-rank adaptation (LoRA) or full finetuning on downstream tasks
(feeding tube classification and patients' clinical characteristics), compared to training without
pretraining. Pretraining is tracked with the validation loss, RankMe and linear probing.

The patient data used in the manuscript cannot be shared; this repository lets you pretrain and finetune the same models on your own data, and documents the exact settings of the manuscript.

## Installation

```bash
conda env create -f environment.yaml
conda activate ssl-rt
```

The environment also installs this repository in editable mode as the Python package `src`
(`pip install -e .`), so `from src.models... import ...` works from anywhere, e.g. in notebooks.
The scripts do not depend on it: they add the repository root to the Python path themselves, so
`python src/<script>.py` also works without installing the package.

## Repository structure

```
src/
  pretrain.py              SimMIM pretraining
  finetune.py              full finetuning and linear probing of a pretrained encoder
  finetune_lora.py         LoRA finetuning of a pretrained encoder
  train_supervised.py      no-pretraining baseline (ViT trained from scratch)
  compute_rankme.py        RankMe of a pretraining checkpoint
  inference/               test-set evaluation of saved finetuning / LoRA / supervised models
  data/                    datamodules (pretraining, downstream tasks) and SimMIM masking
  models/                  Lightning modules and the 3D ViT (models/components/vit.py)
  utils/                   training loop, checkpoint loading, metrics, callbacks
configs/
  pretrain.yaml, finetune.yaml, finetune_lora.yaml, train_supervised.yaml, compute_rankme.yaml
  experiment/              manuscript settings: pretrain/, feeding_tube/, clinical/
  task/                    clinical characteristic (sex, t_stage, n_stage, tumor_site, age)
  data/, model/, callbacks/, trainer/, logger/, paths/
scripts/measure_flops.py   FLOPs per training step (for PF-days)
data_analysis/             notebooks for the figures and mixed-effects models of the manuscript
```

## Data

Paths are set with environment variables (or in `configs/paths/default.yaml`):

| Variable | Content |
|---|---|
| `PRETRAIN_DATA_DIR` | unlabelled CBCTs and `pretrain_cbcts.csv` (columns `split`, `cbct_path_npy`) |
| `FINETUNE_DATA_DIR` | labelled CBCTs and the train/test csv files of the downstream tasks |
| `PRETRAINED_DIR` | pretrained checkpoints, as `<model_name>/epoch_epoch=<NNN>.ckpt` |

All volumes are resampled to 1 x 1 x 2.5 mm, clamped to [-1000, 1000] HU, rescaled to [-1, 1] and
cropped to 256 x 224 x 64 around the body. In the manuscript this preprocessing was done offline and
the volumes stored as `.npy` (`read_numpy: True`); with `read_numpy: False` the same preprocessing
runs on the fly from the raw images. The csv columns expected by each datamodule are documented in
[`src/data/pretrain_datamodule.py`](src/data/pretrain_datamodule.py) and
[`src/data/finetune_datamodule.py`](src/data/finetune_datamodule.py); the csv file names are set in
`configs/data/`.

## Reproducing the manuscript experiments

Each experiment config fixes the settings used in the manuscript; any value can be overridden from
the command line, and `-m` runs a sweep.

```bash
# SimMIM pretraining (100 epochs, a checkpoint per epoch)
python src/pretrain.py experiment=pretrain/vit_s

# pretraining tracking: RankMe and linear probing of every pretraining epoch
python src/compute_rankme.py -m model_name=vit-s pretrained_epoch="range(0,100)" csv_file=<images.csv>
python src/finetune.py -m experiment=feeding_tube/linear_probe model_name=vit-s pretrained_epoch="range(0,100)"

# feeding tube classification: LoRA, full finetuning and no pretraining
python src/finetune_lora.py experiment=feeding_tube/lora model_name=vit-s
python src/finetune.py experiment=feeding_tube/full_finetune model_name=vit-s
python src/train_supervised.py experiment=feeding_tube/supervised model.model_name=vit_small

# clinical characteristics (task = sex | t_stage | n_stage | tumor_site; age has its own experiments)
python src/finetune_lora.py experiment=clinical/lora task=t_stage model_name=vit-s
python src/finetune.py experiment=clinical/full_finetune_age model_name=vit-s
```

The ablations (pretraining length and annotation budget, five replicates each) are sweeps over
`pretrained_epoch`, `data.train_subsplit` and `data.seed`; the exact commands are given at the top
of each file in [`configs/experiment/`](configs/experiment/).

Every finetuning run evaluates the model on the test set at the end of training (the results
reported in the manuscript). Saved models can be re-evaluated with the scripts in `src/inference/`,
using the same experiment and overrides as for training:

```bash
python src/inference/lora.py experiment=feeding_tube/lora model_name=vit-s \
    lora_checkpoint_path=logs/finetune_lora/runs/<run>/lora_checkpoints
```

Pretraining compute (FLOPs per step, for PF-days) is measured with
`python scripts/measure_flops.py --model vit_small --batch-size 30`.

Experiments in the manuscript were logged with Weights & Biases (`logger=wandb`); the default
logger is a local CSV logger.

## License

The code in this repository is licensed under the Apache License 2.0 — see LICENSE. 
