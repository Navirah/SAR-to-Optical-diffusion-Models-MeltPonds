from __future__ import annotations

import os
import ast
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy import ndimage
import torch

# Configuration

@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 256
    stride: int = 256
    mpf_threshold: float = 0.10
    min_cluster_area: int = 100
    blank_threshold: float = 1e-6

# MPF utilities

def load_mpf(path: str, var: str = "mpf") -> np.ndarray:
    """Load MPF array from NetCDF."""
    with xr.open_dataset(path) as ds:
        return ds[var].values.squeeze()


def find_mpf_clusters(
    mpf: np.ndarray,
    threshold: float,
    min_area: int
) -> List[Tuple[int, int, int, int]]:
    """
    Identify connected MPF clusters and return bounding boxes.
    Boxes are (min_row, min_col, max_row, max_col).
    """
    mask = mpf > threshold
    labeled, _ = ndimage.label(mask)
    slices = ndimage.find_objects(labeled)

    bboxes = []
    for idx, slc in enumerate(slices):
        region = labeled[slc] == (idx + 1)
        if region.sum() >= min_area:
            r0, r1 = slc[0].start, slc[0].stop
            c0, c1 = slc[1].start, slc[1].stop
            bboxes.append((r0, c0, r1, c1))
    return bboxes

# Raster loading

def load_s1_stack(base_path: str) -> np.ndarray:
    """Load Sentinel-1 HH/HV stack."""
    hh_path = os.path.join(base_path, "Sigma0_HH_db_corr028_m.img") #based on processed data
    hv_path = os.path.join(base_path, "Sigma0_HV_db_corr024_m.img")

    with rasterio.open(hh_path) as hh, rasterio.open(hv_path) as hv:
        return np.stack([hh.read(1), hv.read(1)], axis=0)


def load_s2_stack(base_path: str) -> np.ndarray:
    """Load Sentinel-2 B2, B3, B4, B8 stack."""
    bands = ["B2", "B3", "B4", "B8"]
    arrays = []
    for b in bands:
        with rasterio.open(os.path.join(base_path, f"{b}.img")) as src:
            arrays.append(src.read(1))
    return np.stack(arrays, axis=0)

# Cropping & patching

def extract_crops(
    s1: np.ndarray,
    s2: np.ndarray,
    mpf: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]]
):
    """Crop S1, S2 and MPF using bounding boxes."""
    for r0, c0, r1, c1 in bboxes:
        yield (
            s1[:, r0:r1, c0:c1],
            s2[:, r0:r1, c0:c1],
            mpf[r0:r1, c0:c1][None, ...],
        )


def extract_patches(
    s1_crop: np.ndarray,
    s2_crop: np.ndarray,
    mpf_crop: np.ndarray,
    cfg: PatchConfig
):
    """Extract fixed-size patches from a crop."""
    h, w = s1_crop.shape[1:]

    for r in range(0, h - cfg.patch_size + 1, cfg.stride):
        for c in range(0, w - cfg.patch_size + 1, cfg.stride):
            s1_p = s1_crop[:, r:r+cfg.patch_size, c:c+cfg.patch_size]
            if np.all(np.abs(s1_p) < cfg.blank_threshold):
                continue

            yield (
                s1_p,
                s2_crop[:, r:r+cfg.patch_size, c:c+cfg.patch_size],
                mpf_crop[:, r:r+cfg.patch_size, c:c+cfg.patch_size],
            )

# Dataset assembly

def build_patch_dataset(
    csv_path: str,
    cfg: PatchConfig,
):
    """Main dataset construction routine."""
    df = pd.read_csv(csv_path)
    df["collocated_folders"] = df["collocated_folders"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    s1_all, s2_all, mpf_all, dates = [], [], [], []

    for _, row in df.iterrows():
        mpf = load_mpf(row["file"])
        date = int(os.path.basename(row["file"]).split("_")[0])

        bboxes = find_mpf_clusters(
            mpf,
            threshold=cfg.mpf_threshold,
            min_area=cfg.min_cluster_area,
        )

        for base_path in row["collocated_folders"]:
            s1 = load_s1_stack(base_path)
            s2 = load_s2_stack(base_path)

            for s1_crop, s2_crop, mpf_crop in extract_crops(s1, s2, mpf, bboxes):
                for s1_p, s2_p, mpf_p in extract_patches(
                    s1_crop, s2_crop, mpf_crop, cfg
                ):
                    s1_all.append(s1_p)
                    s2_all.append(s2_p)
                    mpf_all.append(mpf_p)
                    dates.append(date)

    return (
        np.asarray(s1_all),
        np.asarray(s2_all),
        np.asarray(mpf_all),
        np.asarray(dates),
    )

def save_patch_dataset(path: str, s1, s2, mpf, dates):
    """Save dataset to compressed NPZ."""
    np.savez_compressed(
        path,
        s1=s1,
        s2=s2,
        mpf=mpf,
        date=dates,
    )