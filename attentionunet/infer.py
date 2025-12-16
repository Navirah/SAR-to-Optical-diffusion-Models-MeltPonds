"""
Simple inference script for S1 → S2 using a trained diffusion model.

Runs conditional sampling on a TFRecord test set with:
- Karras noise schedule (rho=7, 40 steps)
- Heun sampler (EDM-style)
- Lightning diffusion wrapper + UNet checkpoint

Outputs:
  preds_s1_to_s2_karras_heun_simple.npz  -> predictions, inputs, ground truth
  metrics_summary.json                   -> aggregated metrics
  metrics_summary.csv                    -> metrics in CSV format
  previews/batch_00000.png               -> optional visual previews
"""

import os
import sys
from pathlib import Path
import importlib.util
import json
import csv

import numpy as np
import torch
from tqdm.auto import tqdm

#experiment configuration

from configs.eval_config import (
    CKPT,
    TFRECORD_ROOT,
    OUT_DIR,
    CONFIG_PATH,
    N_STEPS,
    SIGMA_MIN,
    SIGMA_MAX,
    RHO,
    DEVICE,
    PREVIEW_EVERY,
    PREVIEW_SAMPLES,
)

#repo imports

from lightning import utils as lightning_utils
from sampling.k_sampling import get_sigmas_karras, sample_heun

torch.backends.cudnn.benchmark = True


#helpers

def _move_like(t, ref_param):
    """Move tensors to the same device and dtype as a reference parameter."""
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
            non_blocking=True
        )
    if isinstance(t, (list, tuple)):
        return type(t)(
            _move_like(x, ref_param) if torch.is_tensor(x) else x
            for x in t
        )
    return t


def _unwrap_cond_image(cond):
    """Extract the conditioning image tensor."""
    if isinstance(cond, (list, tuple)) and torch.is_tensor(cond[0]):
        return cond[0]
    if torch.is_tensor(cond):
        return cond
    raise RuntimeError(f"Unexpected cond type: {type(cond)}")


#visualisation helpers

def _percentile_norm(x, pmin=2, pmax=98):
    if x.ndim == 2:
        x = x[..., None]
    y = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        ch = x[..., c]
        m = np.isfinite(ch)
        if not m.any():
            y[..., c] = 0.0
            continue
        lo = np.percentile(ch[m], pmin)
        hi = np.percentile(ch[m], pmax)
        y[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if y.shape[-1] == 1:
        y = np.repeat(y, 3, axis=-1)
    return y


def _s2_to_rgb_chw(s2):
    rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)
    return _percentile_norm(rgb)


def _s1_to_viz_chw(s1):
    vv, vh = s1[0], s1[1]
    rgb = np.stack([vv, vh, vv], axis=-1)
    return _percentile_norm(rgb)


#metrics

def _safe_flat_pair(pred, true):
    p = pred.ravel()
    t = true.ravel()
    m = np.isfinite(p) & np.isfinite(t)
    return (p[m], t[m]) if m.any() else (None, None)


def mae(p, t):
    p, t = _safe_flat_pair(p, t)
    return float(np.mean(np.abs(p - t))) if p is not None else float("nan")


def mse(p, t):
    p, t = _safe_flat_pair(p, t)
    return float(np.mean((p - t) ** 2)) if p is not None else float("nan")


def rmse(p, t):
    return float(np.sqrt(mse(p, t)))


def psnr(p, t):
    err = mse(p, t)
    if err <= 0:
        return float("inf")
    dr = np.nanmax(t) - np.nanmin(t)
    return float(20 * np.log10(dr) - 10 * np.log10(err))


def pearsonr(p, t):
    p, t = _safe_flat_pair(p, t)
    if p is None:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def compute_metrics(preds, truths, var_names=None):
    metrics = {"overall": {}, "per_band": []}

    metrics["overall"].update(
        mae=mae(preds, truths),
        rmse=rmse(preds, truths),
        mse=mse(preds, truths),
        psnr=psnr(preds, truths),
        pearson_r=pearsonr(preds, truths),
    )

    for c in range(preds.shape[1]):
        name = var_names[c] if var_names else f"band_{c}"
        metrics["per_band"].append(
            dict(
                band=name,
                mae=mae(preds[:, c], truths[:, c]),
                rmse=rmse(preds[:, c], truths[:, c]),
                mse=mse(preds[:, c], truths[:, c]),
                psnr=psnr(preds[:, c], truths[:, c]),
                pearson_r=pearsonr(preds[:, c], truths[:, c]),
            )
        )
    return metrics


def save_metrics(metrics, out_dir: Path):
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scope", "band", "mae", "rmse", "mse", "psnr", "pearson_r"]
        )
        o = metrics["overall"]
        writer.writerow(
            ["overall", "", o["mae"], o["rmse"], o["mse"], o["psnr"], o["pearson_r"]]
        )
        for b in metrics["per_band"]:
            writer.writerow(
                ["per_band", b["band"], b["mae"], b["rmse"], b["mse"], b["psnr"], b["pearson_r"]]
            )


#config loading

def _load_config_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("user_config", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


#main entry point

def main():
    config = _load_config_from_path(CONFIG_PATH)

    config.tfrecord = True
    config.data.tfrecord_root = TFRECORD_ROOT
    config.training.batch_size = 2
    config.precision = "32"

    model = lightning_utils.build_model(config, str(CKPT))
    model = model.to(DEVICE).eval()

    denoiser = getattr(model, "model", model)
    ref_param = next(denoiser.parameters())

    test_dl = lightning_utils.build_dataloaders_test(
        config, lambda x: x, num_workers=4
    )

    sigmas = get_sigmas_karras(
        n=N_STEPS,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        rho=RHO,
        device=ref_param.device,
    ).to(dtype=ref_param.dtype)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "previews").mkdir(exist_ok=True)

    all_preds, all_conds, all_truths = [], [], []

    with torch.no_grad():
        for _, (cond, truth) in enumerate(tqdm(test_dl, desc="Sampling")):
            all_truths.append(truth.cpu().float().numpy())

            cond = _move_like(cond, ref_param)
            cond_img = _unwrap_cond_image(cond)
            B, _, H, W = cond_img.shape

            x = torch.randn(
                B,
                truth.shape[1],
                H,
                W,
                device=ref_param.device,
                dtype=ref_param.dtype,
            ) * sigmas[0]

            preds = sample_heun(
                model=denoiser,
                x=x,
                cond=cond,
                sigmas=sigmas,
            )

            all_preds.append(preds.cpu().float().numpy())
            all_conds.append(cond_img.cpu().float().numpy())

    preds = np.vstack(all_preds)
    conds = np.vstack(all_conds)
    truths = np.vstack(all_truths)

    np.savez_compressed(
        OUT_DIR / "preds_s1_to_s2_karras_heun_simple.npz",
        preds=preds,
        conds=conds,
        truths=truths,
    )

    var_names = list(config.data.variables[1])
    metrics = compute_metrics(preds, truths, var_names)
    save_metrics(metrics, OUT_DIR)


if __name__ == "__main__":
    main()
