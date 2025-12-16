import os, csv, argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import make_grid, save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from train_x0 import load_dataloaders_from_npz_test
from models.unet_cond_multiscale import UNetCondMultiScale
from diffusion.scheduler_x0 import DDPMX0Scheduler
import utils.helpers as helper 

# Sentinel-2 (train-set stats)
S2_MEAN = torch.tensor([1549.4233, 1361.8755, 1372.6019, 1037.0249], dtype=torch.float32)
S2_STD  = torch.tensor([2861.3311, 2766.7141, 2694.3137, 2237.9646], dtype=torch.float32)

def denorm_s2_z_to_native(x_bchw: torch.Tensor) -> torch.Tensor:
    mean = S2_MEAN.to(x_bchw.device).view(1, 4, 1, 1)
    std  = S2_STD.to(x_bchw.device).view(1, 4, 1, 1)
    return x_bchw * std + mean

#Visualisation helpers
def s1_to_rgb01(s1_tensor: torch.Tensor) -> torch.Tensor:
    s1 = s1_tensor.detach().cpu()
    vv = s1[:, 0:1]
    vh = s1[:, 1:2]
    blue = 0.5 * (vv + vh)
    rgb = torch.cat([vv, vh, blue], dim=1)
    mn = rgb.amin(dim=[2, 3], keepdim=True)
    mx = rgb.amax(dim=[2, 3], keepdim=True)
    return ((rgb - mn) / (mx - mn + 1e-6)).clamp(0, 1)

def s2_to_rgb01(s2_tensor: torch.Tensor) -> torch.Tensor:
    return helper.s2_to_rgb01_z(s2_tensor.detach().cpu()).clamp(0, 1)

#metrics
def spectral_angle_mapper(gt_hwC: np.ndarray, pred_hwC: np.ndarray, eps=1e-8) -> float:
    dot = np.sum(gt_hwC * pred_hwC, axis=-1)
    ngt = np.linalg.norm(gt_hwC, axis=-1)
    npr = np.linalg.norm(pred_hwC, axis=-1)
    cosang = dot / (ngt * npr + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.mean(np.arccos(cosang)))

def safe_data_range(gt_np: np.ndarray) -> float:
    mn, mx = float(gt_np.min()), float(gt_np.max())
    if -1e-6 <= mn and mx <= 1.0 + 1e-6:
        return 1.0
    return max(mx - mn, 1e-6)

#panel saving
def save_panel(preds, gts, s1, global_idx, out_dir,
               panel_every, panel_limit, save_numpy):
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)

    pred_rgb = s2_to_rgb01(preds)
    gt_rgb   = s2_to_rgb01(gts)
    s1_rgb   = s1_to_rgb01(s1)

    B = preds.size(0)
    for b in range(B):
        idx = global_idx + b
        if idx % panel_every != 0 or idx >= panel_limit:
            continue

        grid = make_grid(
            torch.cat([s1_rgb[b:b+1], gt_rgb[b:b+1], pred_rgb[b:b+1]], dim=0),
            nrow=3, padding=2
        )
        save_image(grid, os.path.join(out_dir, "panels", f"sample_{idx:06d}.png"))

        if save_numpy:
            np.save(os.path.join(out_dir, "npy", f"{idx:06d}_s1.npy"), s1[b].cpu().numpy())
            np.save(os.path.join(out_dir, "npy", f"{idx:06d}_gt.npy"), gts[b].cpu().numpy())
            np.save(os.path.join(out_dir, "npy", f"{idx:06d}_pred.npy"), preds[b].cpu().numpy())

#evaluation
@torch.inference_mode()
def evaluate(model, loader, scheduler, device, save_dir,
             save_panels=True, save_numpy=False,
             panel_every=1, panel_limit=999999,
             max_samples=None):

    metrics = {"psnr": [], "ssim": [], "rmse": [], "sam": []}
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["idx", "psnr", "ssim", "rmse", "sam"])

    seen = 0
    global_idx = 0

    for batch in tqdm(loader, desc="Evaluating"):
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        B, C, H, W = s2.shape

        preds = scheduler.sample(
            model,
            cond=s1,
            shape=(B, C, H, W),
            device=device,
            progress=False
        )

        preds_nat = denorm_s2_z_to_native(preds)
        s2_nat    = denorm_s2_z_to_native(s2)

        for b in range(B):
            if max_samples is not None and seen >= max_samples:
                break

            pred = np.moveaxis(preds_nat[b].cpu().numpy(), 0, -1)
            gt   = np.moveaxis(s2_nat[b].cpu().numpy(),    0, -1)

            dr = safe_data_range(gt)
            psnr = peak_signal_noise_ratio(gt, pred, data_range=dr)
            ssim = structural_similarity(gt, pred, channel_axis=2, data_range=dr)
            rmse = np.sqrt(mean_squared_error(gt.reshape(-1), pred.reshape(-1)))
            sam  = spectral_angle_mapper(gt, pred)

            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["rmse"].append(rmse)
            metrics["sam"].append(sam)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([global_idx, psnr, ssim, rmse, sam])

            global_idx += 1
            seen += 1

        if save_panels:
            save_panel(preds, s2, s1, global_idx - B, save_dir,
                       panel_every, panel_limit, save_numpy)

        if max_samples is not None and seen >= max_samples:
            break

    with open(os.path.join(save_dir, "summary_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {np.mean(v):.6f}\n")

    return {k: float(np.mean(v)) for k, v in metrics.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default="test_outputs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_panels", action="store_true")
    parser.add_argument("--save_numpy", action="store_true")
    parser.add_argument("--panel_every", type=int, default=1)
    parser.add_argument("--panel_limit", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetCondMultiScale(
        in_ch=4, cond_ch=2, out_ch=4,
        base_ch=64, ch_mults=(1, 2, 4), t_dim=128,
        out_activation=None
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    scheduler = DDPMX0Scheduler(
        num_timesteps=1000,
        schedule=args.schedule
    ).to(device)

    test_loader, _ = load_dataloaders_from_npz_test(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    evaluate(
        model, test_loader, scheduler, device, args.save_dir,
        save_panels=args.save_panels,
        save_numpy=args.save_numpy,
        panel_every=args.panel_every,
        panel_limit=args.panel_limit,
        max_samples=None if args.max_samples <= 0 else args.max_samples
    )

if __name__ == "__main__":
    main()
