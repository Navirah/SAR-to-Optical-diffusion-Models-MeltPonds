"""
eval_config.py

Centralised configuration for diffusion model evaluation and inference.
This file contains experiment constants only.
"""

from pathlib import Path
import torch

#paths

CKPT = Path(
    "/path/to/outputs/"
    "diffusion_downscaling/multipredict/karras_p_diff_noscale/"
    "chkpts/epoch=71-val_loss=0.5559.ckpt"
)

TFRECORD_ROOT = Path(
    "/path/to/prepared_data/curated/256"
)

OUT_DIR = Path(
    "/path/to/outputs/simple_run_epoch71"
)

CONFIG_PATH = Path(
    "/path/to/config.py"
)

#sampling (Karras / EDM)

N_STEPS = 40
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0

#runtime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#preview and visualisation

PREVIEW_EVERY = 2     # save preview every N batches
PREVIEW_SAMPLES = 4   # rows per preview image
