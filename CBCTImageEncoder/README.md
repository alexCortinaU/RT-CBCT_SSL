# CBCTImageEncoder

Pretrained 3D Vision Transformer encoders (tiny / small / base) for CBCT (Cone-Beam CT) volumes, trained with SimMIM self-supervised pretraining. This repo is the minimal code needed to load the encoders and use them. The training/research code to develop them lives in [RT-CBCT_SSL](../).

> **Disclaimer**: this code and the associated pretrained weights are released for
> **research use only**. They are **not a medical device** and have **not been
> validated for clinical use**. Do not use them to inform clinical decisions.

## Install

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `pyyaml`, `torchio`. `peft` is only needed for
`scripts/lora_finetune_example.py`; `pandas` only for the two `*_finetune_example.py`
scripts (reading a labels CSV).

## Weights

Not included in this repository. Download them and place them under `checkpoints/` — see
[`checkpoints/README.md`](checkpoints/README.md) for the exact layout and the [download link](https://doi.org/doi:10.60612/DATADK/BTAQSF).

## Quickstart

```python
from src import load_pretrained

model = load_pretrained("small", checkpoints_dir="checkpoints")  # or "tiny" / "base"

# x: a preprocessed CBCT volume, shape (batch, 1, 256, 224, 64) -- see "Expected input" below
patch_tokens, cls_token = model(x, return_cls_token=True)
# cls_token: (batch, embed_dim) -- a single embedding vector per volume
# patch_tokens: (batch, 896, embed_dim) -- one token per 16^3 patch, if you need
#               finer-grained (per-region) features instead of one pooled vector

# Or, if you only need the patch tokens:
patch_tokens = model(x)
```

`embed_dim` is 192 / 384 / 768 for tiny / small / base respectively.

## Expected input

- **Volume size**: 256 × 224 × 64 voxels
- **Voxel spacing**: 1 × 1 × 2.5 mm
- **Patch size**: 16³ (→ a 16×14×4 = 896-token patch grid)
- **Intensity**: clamped to [-1000, 1000] HU, then rescaled to [-1, 1]
- **Crop**: centered on the volume (no body-segmentation-aware cropping)

`scripts/inference.py` implements exactly this preprocessing (reorient → resample →
centered crop/pad → clamp → rescale) for a folder of NIfTI files, via `torchio`. It's a
**simple centered crop** — no attempt to segment or center on a specific anatomical
region first. If your scans have a lot of surrounding table/immobilization
mask/background, or the anatomy of interest isn't already roughly centered in the field
of view, a plain center crop may cut off relevant anatomy or leave it off-center after
resizing. **If that matters for your data, preprocess it offline first** — crop around
the body/anatomy of interest yourself (with whatever segmentation approach fits your
data), then point `--data_dir` at the already-cropped volumes; the center crop this
script does will then be close to a no-op.

## Scripts

### `scripts/inference.py` — get embeddings for a folder of scans

```bash
python scripts/inference.py --data_dir /path/to/nifti --output_dir /path/to/out \
    --model_size small --checkpoints_dir checkpoints
```

Reads every `.nii.gz` under `--data_dir` (via a small `NiftiCBCTDataset` +
`DataLoader`, preprocessing each volume "online" — per-item, in `__getitem__`), and
saves one `<case>.npy` CLS embedding per scan into `--output_dir`.

### `scripts/lora_finetune_example.py` — LoRA finetuning

```bash
python scripts/lora_finetune_example.py --data_dir /path/to/nifti --labels_csv labels.csv \
    --model_size small --checkpoints_dir checkpoints --adapter_out lora_adapter/
```

`labels_csv`: two columns, `filename` (a `.nii.gz` under `--data_dir`) and `label` (an
integer class index). Freezes the pretrained encoder, wraps it with a
[peft](https://github.com/huggingface/peft) LoRA adapter (`target_modules="all-linear"`)
plus a trainable linear head, and runs a short training loop.

**Saving the adapter** (only the small LoRA delta weights + head, a few MB — not the
base encoder):
```python
model.save_pretrained("lora_adapter/")
```

**Loading it back** (in a fresh process — rebuild the same frozen base model first, then
attach the adapter):
```python
from src import load_pretrained
from peft import PeftModel

encoder = load_pretrained("small", "checkpoints")
base_model = ClassifierModel(encoder, num_classes=2)  # see lora_finetune_example.py
model = PeftModel.from_pretrained(base_model, "lora_adapter/")
```

(`scripts/lora_finetune_example.py` has a ready-made `load_finetuned(...)` helper that
does exactly this.)

### `scripts/full_finetune_example.py` — full finetuning

Same shape as the LoRA example, without `peft` — every encoder weight is trainable, not
just a small adapter. Saves/loads a plain `state_dict` via `torch.save`/`load_state_dict`
(see the script's own `load_finetuned(...)` helper).

## Repository structure

```
CBCTImageEncoder/
├── README.md
├── LICENSE
├── requirements.txt
├── src/                          # the model itself -- load_pretrained(), build_model(), preprocessing
│   ├── model.py
│   ├── pos_embed.py
│   └── preprocessing.py
├── MODEL_CARD.md                 # weights' license, training data, intended use
├── checkpoints/
│   ├── README.md                 # weights download instructions
│   ├── vit_tiny_config.yaml
│   ├── vit_small_config.yaml
│   └── vit_base_config.yaml      # *_weights.ckpt not included -- see checkpoints/README.md
└── scripts/
    ├── inference.py
    ├── lora_finetune_example.py
    └── full_finetune_example.py
```

## License

The **code** in this repository is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE). The **pretrained weights** (distributed separately, not in this
repo) are licensed separately under CC BY-NC-SA 4.0 (non-commercial) — see
[MODEL_CARD.md](MODEL_CARD.md),
which also covers training data and intended use.
