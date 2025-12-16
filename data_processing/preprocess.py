from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple, Dict, Callable, Optional

import numpy as np
import pandas as pd

# Channel utilities

def _ensure_channels_last(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is in (N, H, W, C) format.

    Accepts:
      - (N, H, W, C)
      - (N, C, H, W)

    :param arr: [np.ndarray] 4D array
    :return: [np.ndarray] Array in channels-last format
    """
    assert arr.ndim == 4, f"Expected 4D array, got shape {arr.shape}"

    plausible_channels = {1, 2, 3, 4, 8, 10, 12, 13}

    # Already channels-last
    if arr.shape[-1] in plausible_channels:
        return arr

    # Channels-first
    if arr.shape[1] in plausible_channels:
        return np.transpose(arr, (0, 2, 3, 1))

    raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")

# Normalization statistics

def _compute_channel_stats(
    arr: np.ndarray,
    method: str,
) -> Tuple[Dict, Callable, Callable]:
    """
    Compute per-channel normalization statistics on TRAIN data only.

    :param arr: [np.ndarray] Shape (N, H, W, C), channels-last
    :param method: [str] "zscore" or "minmax"
    :return:
        - stats: dict with fitted parameters
        - norm_fn: callable for normalization
        - denorm_fn: callable for inverse transform
    """
    assert arr.ndim == 4, f"Expected (N,H,W,C), got {arr.shape}"

    C = arr.shape[-1]
    flat = arr.reshape(-1, C).astype(np.float32)

    if method == "zscore":
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        std = np.where(std == 0.0, 1.0, std)

        def norm_fn(x):
            return (x - mean) / std

        def denorm_fn(x):
            return x * std + mean

        stats = {
            "method": "zscore",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    elif method == "minmax":
        vmin = np.nanmin(flat, axis=0)
        vmax = np.nanmax(flat, axis=0)
        rng = vmax - vmin
        rng = np.where(rng == 0.0, 1.0, rng)

        def norm_fn(x):
            return (x - vmin) / rng

        def denorm_fn(x):
            return x * rng + vmin

        stats = {
            "method": "minmax",
            "min": vmin.tolist(),
            "max": vmax.tolist(),
        }

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return stats, norm_fn, denorm_fn

# Temporal splitting

def _split_by_date_countaware(
    dates_ordinal: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    gap_days: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal split by date, count-aware and leakage-safe.

    - All samples from a date stay in the same split
    - Split ratios are based on sample counts, not number of dates

    :param dates_ordinal: [np.ndarray] Ordinal dates (shape N)
    :param val_ratio: [float] Validation fraction
    :param test_ratio: [float] Test fraction
    :param gap_days: [int] Optional temporal gap between splits
    :return: train_idx, val_idx, test_idx
    """
    N = len(dates_ordinal)
    idx_all = np.arange(N)

    uniq_dates, inv = np.unique(dates_ordinal, return_inverse=True)
    counts = np.bincount(inv)
    cum = np.cumsum(counts)

    n_train = int(np.floor(N * (1.0 - val_ratio - test_ratio)))
    n_val_end = int(np.floor(N * (1.0 - test_ratio)))

    train_last = np.searchsorted(cum, n_train, side="right") - 1
    train_last = max(train_last, 0)

    val_start = train_last + 1 + gap_days
    val_end = np.searchsorted(cum, n_val_end, side="right") - 1
    val_end = max(val_end, val_start)

    test_start = val_end + 1 + gap_days

    train_dates = uniq_dates[: train_last + 1]
    val_dates = uniq_dates[val_start : val_end + 1]
    test_dates = uniq_dates[test_start :]

    train_mask = np.isin(dates_ordinal, train_dates)
    val_mask = np.isin(dates_ordinal, val_dates)
    test_mask = np.isin(dates_ordinal, test_dates)

    return idx_all[train_mask], idx_all[val_mask], idx_all[test_mask]

# Main preprocessing routine

def preprocess_dataframe(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    s1_method: str = "zscore",
    s2_method: str = "zscore",
    mpf_method: str = "minmax",
    use_temporal_split: bool = True,
    gap_days: int = 0,
    stats_out_path: Optional[str] = None,
):
    """
    Preprocess patch-level dataset.

    Expected DataFrame columns:
      - s1   : np.ndarray (H,W,2) or (2,H,W)
      - s2   : np.ndarray (H,W,4) or (4,H,W)
      - mpf  : np.ndarray (H,W,1) [optional]
      - date : int YYYYMMDD

    :return:
        processed: dict with splits -> (s1, s2, mpf, dates)
        stats: fitted normalization statistics
        denormers: inverse transforms
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    dates_ordinal = df["date"].map(lambda d: d.toordinal()).to_numpy()

    s1_all = _ensure_channels_last(np.stack(df["s1"].to_numpy()).astype(np.float32))
    s2_all = _ensure_channels_last(np.stack(df["s2"].to_numpy()).astype(np.float32))
    mpf_all = (
        _ensure_channels_last(np.stack(df["mpf"].to_numpy()).astype(np.float32))
        if "mpf" in df.columns
        else None
    )

    if use_temporal_split:
        train_idx, val_idx, test_idx = _split_by_date_countaware(
            dates_ordinal,
            val_ratio,
            test_ratio,
            gap_days,
        )
    else:
        raise NotImplementedError("Random split disabled to avoid leakage.")

    print(
        f"Split sizes → train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    s1_stats, s1_norm, s1_denorm = _compute_channel_stats(
        s1_all[train_idx], s1_method
    )
    s2_stats, s2_norm, s2_denorm = _compute_channel_stats(
        s2_all[train_idx], s2_method
    )

    if mpf_all is not None:
        mpf_stats, mpf_norm, mpf_denorm = _compute_channel_stats(
            mpf_all[train_idx], mpf_method
        )
    else:
        mpf_stats = mpf_norm = mpf_denorm = None

    def _apply(idx):
        return (
            s1_norm(s1_all[idx]),
            s2_norm(s2_all[idx]),
            mpf_norm(mpf_all[idx]) if mpf_all is not None else None,
            dates_ordinal[idx],
        )

    processed = {
        "train": _apply(train_idx),
        "val": _apply(val_idx),
        "test": _apply(test_idx),
    }

    stats = {
        "s1": s1_stats,
        "s2": s2_stats,
        "mpf": mpf_stats,
        "split": {
            "mode": "temporal",
            "gap_days": gap_days,
        },
    }

    if stats_out_path:
        Path(stats_out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_out_path, "w") as f:
            json.dump(stats, f, indent=2)

    denormers = {
        "s1": s1_denorm,
        "s2": s2_denorm,
        "mpf": mpf_denorm,
    }

    return processed, stats, denormers

# Saving utilities

def save_processed_to_npz(
    processed: Dict,
    out_dir: str,
):
    """
    Save processed dataset splits to compressed NPZ files.

    :param processed: dict from preprocess_dataframe
    :param out_dir: [str] Output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    for split, (s1, s2, mpf, dates) in processed.items():
        path = os.path.join(out_dir, f"{split}.npz")
        np.savez_compressed(
            path,
            s1=s1.astype(np.float32),
            s2=s2.astype(np.float32),
            mpf=mpf.astype(np.float32) if mpf is not None else np.array([]),
            dates=dates.astype(np.int64),
        )
        print(f"[SAVE] {split} → {path}")


def main():
    """
    End-to-end preprocessing entry point.
    """
    npz_path = "patch_dataset.npz"
    out_dir = "prepared_data_new"
    stats_path = "artifacts/train_stats.json"

    data = np.load(npz_path)
    df = pd.DataFrame({k: list(data[k]) for k in data.files})

    processed, _, _ = preprocess_dataframe(
        df,
        val_ratio=0.15,
        test_ratio=0.15,
        s1_method="zscore",
        s2_method="zscore",
        mpf_method="minmax",
        use_temporal_split=True,
        gap_days=0,
        stats_out_path=stats_path,
    )

    save_processed_to_npz(processed, out_dir)


if __name__ == "__main__":
    main()
