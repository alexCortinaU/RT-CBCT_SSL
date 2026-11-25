# Checkpoints

`vit_{tiny,small,base}_config.yaml` ship with this repo (tracked in git). The matching
`vit_{tiny,small,base}_weights.ckpt` do **not** — they're hosted separately:

[Cortina Uribe, Alejandro. 2026. “Supporting Data for: Impact of Pretraining Length, Model Size, and Annotation Budget on Downstream Clinical Tasks of Foundation Models in Radiotherapy.” DeiC Dataverse. https://doi.org/doi:10.60612/DATADK/BTAQSF](https://doi.org/doi:10.60612/DATADK/BTAQSF)

> Note: The DeiC Dataverse repository might be inaccesible at the moment. Please reach out via this Github repo to receive the models' weights.

See [`MODEL_CARD.md`](../MODEL_CARD.md) for the weights' license (CC BY-NC-SA 4.0, non-commercial,
separate from this repo's own Apache 2.0 code license), training data description, and intended use.

After downloading, place the weight files directly alongside the config files:

```
checkpoints/
├── vit_tiny_config.yaml    (in this repo)
├── vit_tiny_weights.ckpt   (download separately)
├── vit_small_config.yaml   (in this repo)
├── vit_small_weights.ckpt  (download separately)
├── vit_base_config.yaml    (in this repo)
└── vit_base_weights.ckpt   (download separately)
```

Each `weights.ckpt` is a plain PyTorch `state_dict` (not a Lightning checkpoint) — load
it directly with `src.load_pretrained("small")` (or `"tiny"` / `"base"`), which reads
the matching `config.yaml` for the architecture and `weights.ckpt` for the weights.
