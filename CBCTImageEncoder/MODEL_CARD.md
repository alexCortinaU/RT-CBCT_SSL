# Model Card: CBCTImageEncoder pretrained weights

This card covers the **pretrained weights** (`vit_{tiny,small,base}_weights.ckpt`,
distributed separately — see `checkpoints/README.md`). The code in this repository that
loads and uses them is licensed separately (Apache 2.0, see `LICENSE`).

## Model details

- **Architecture**: 3D Vision Transformer encoder, patch size 16³, 12 transformer
  blocks, sin-cos (non-learnable) position embeddings, a learnable CLS token.
- **Sizes**:

  | Size | `embed_dim` | heads | params (encoder only) |
  |---|---|---|---|
  | tiny | 192 | 3 | 6,297,600 |
  | small | 384 | 6 | 23,212,032 |
  | base | 768 | 12 | 88,891,392 |

- **Pretraining method**: [SimMIM](https://arxiv.org/abs/2111.09886) (self-supervised
  masked image modeling) — the encoder learns by reconstructing randomly masked patches
  of the input volume. Trained for 100 epochs. No labels were used for pretraining.
- **Training input**: 256×224×64 voxel volumes at 1×1×2.5mm spacing (see the main
  README's "Expected input" section — this is also what the released encoder expects at
  inference time).

## Training data

- See our pre-print for more detailed information on the training cohort: [A. Cortina Uribe, S. H. Poulsen, I. R. Vogelius, and J. Petersen, "Impact of pretraining length, model size, and annotation budget on downstream clinical tasks of foundation models in radiotherapy," arXiv preprint, 2026.arXiv: abs/2611.22896](https://arxiv.org/abs/2611.22896).
- 152,323 3D CBCT (Cone-Beam CT) volumes acquired during radiotherapy treatment courses, from 7,365 patients treated with image-guided RT for lymphoma, head and neck,esophageal and lung cancer.
- Image acquisition period: 2009 - 2023
- Acquisition machines:
    - With their on-board imaging system (OBI): Varian Clinac iX, Varian Clinac 2300iX, Varian Clinac 2100C, Varian Clinac 2300C.
    - Varian TrueBeam, Varian TrueBeam Hypersight
    - Varian Halcyon

## Intended use

- **Intended**: a general-purpose, reusable CBCT volume encoder — extracting embeddings
  (CLS token or patch tokens) for downstream research tasks (classification, regression,
  retrieval, etc.) via linear probing, LoRA, or full finetuning, on CBCT data or similar. Research use only.
- **Out of scope / not intended**:
  - **Clinical use of any kind.** This model is **not a medical device**, has **not**
    been validated for clinical use, and must not be used to inform diagnosis,
    treatment, or any other clinical decision.
  - Anatomical regions or acquisition protocols substantially different from CBCT
    radiotherapy imaging — the encoder was not trained on, and has not
    been evaluated on, other body sites, imaging modalities, or scanner types.
  - Any use that attempts to re-identify patients from whom the training data was
    derived.

## License

The pretrained weights are released under
[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(Attribution-NonCommercial-ShareAlike 4.0 International). You are free to use, share, and adapt
the weights for **non-commercial purposes only**, provided you give appropriate
attribution. Commercial use requires separate permission. See the license text at the
link above for the full terms.

## Disclaimer

This model and its weights are provided "as is", for research purposes only, without
warranty of any kind. They are not a medical device and have not been evaluated or
approved by any regulatory authority for clinical or diagnostic use.
