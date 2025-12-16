# SAR-to-Optical Diffusion Models

This repository contains the code for my MSc thesis on **conditional diffusion models**
for translating **Sentinel-1 SAR** imagery into **Sentinel-2 optical-like** representations,
with a focus on melt-pond monitoring.

The project explores different UNet-based diffusion formulations and compares
their stability, visual quality, and training behaviour.

---

## What this code does

- Learns a conditional mapping from **Sentinel-1 (HH, HV)** to **Sentinel-2 (B2, B3, B4, B8)**
- Uses **diffusion models** rather than direct regression
- Supports both **x₀ prediction** and **ε prediction**
- Includes training, inference, and evaluation pipelines

> This repository assumes that Sentinel-1 and Sentinel-2 data are already
> spatially and temporally aligned.  
> Data colocation is **not** part of this codebase.

---

## Folder structure (key parts)

```text
.
├── attentionunet/     # Model A: attention-based conditional UNet
├── ddpm/              # Model B: DDPM-style UNet (x0 and eps prediction)
├── data_processing/   # Patching and Normalisation
└── notebooks/         # helpers for visualisation and patching
```

---

## Models

Two main model families are implemented:

### Model A — Attention UNet (EDM-style)
- Conditional UNet with attention blocks
- Trained using an EDM / Karras-style noise schedule
- Higher model capacity, less stable training

```text
attentionunet/
├── configs/        # model and training configs
├── data/           # dataset handling and scaling
├── lightning/      # Lightning modules and callbacks
├── sampling/       # diffusion samplers
├── utils/          # helpers and visualisation
├── variables/      # variable definitions and mappings
├── yang_cm/        # core UNet backbone
├── adapter.py      # adapter between backbone and diffusion
├── experiment.py   # experiment setup and state
├── train.py        # training entry point
└── infer.py        # inference and evaluation
```

The `attentionunet/` implementation is based on code originally developed by
**Alex Soulis (UCL)** and **Christian Au** [Au, C., Tsamados, M., Manescu, P., and Takao, S. (2024). Arisgan: Extreme super-resolution
of arctic surface imagery using generative adversarial networks. Frontiers in Remote Sensing,
5:1417417.]

The code has been adapted, refactored, and extended for this project,
including changes to conditioning, diffusion setup, training loops,
and evaluation.

---

### Model B — Lightweight UNet (DDPM-style)
- Compact conditional UNet
- Supports **x₀ prediction** and **ε prediction**

```text
ddpm/
├── datasets/        # dataset loading and preprocessing
├── diffusion/       # DDPM scheduler and noise utilities
├── models/          # UNet architectures
├── training/        # training loops and losses
├── utils/           # helpers and visualisation
├── train_x0.py      # training script for x₀ prediction
├── train_eps.py     # training script for ε prediction
└── eval.py          # evaluation and inference
```
---

## Getting started

The code is organised around two independent model implementations:
- `attentionunet/` (Model A)
- `ddpm/` (Model B)

Each directory contains its own training and evaluation entry points.

Training and evaluation scripts assume:
- a working PyTorch environment
- pre-aligned Sentinel-1 / Sentinel-2 datasets
- configuration files edited to point to local data paths

---

## Intended use

This repository is intended for:
- research and experimentation with diffusion models for Earth observation
- reproducibility of the MSc thesis experiments
- educational reference for conditional diffusion models

It is **not** intended as a production-ready system.

---

## Not included

This repository does not include:
- Sentinel-1 / Sentinel-2 scene search or colocation
- atmospheric correction or radiometric calibration
- melt pond fraction retrieval or thresholding

These steps are described in the thesis but are outside the scope of this codebase.

---

## Citation

If you use this code or build upon it, please cite the accompanying MSc thesis.


