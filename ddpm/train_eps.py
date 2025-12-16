import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from datasets.patch_dataset import PatchDatasetS1S2
from models.unet_cond_multiscale import UNetCondMultiScale
from diffusion.scheduler_eps import DDPMSchedulerEPS
from training.loss_eps import loss_batch_eps
from utils.emission import EmissionsTrackerSession

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

#load data
def _load_split_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["s1"], d["s2"]


def load_dataloaders_from_npz(out_dir, batch_size, num_workers):
    splits = {}
    for split in ["train", "val", "test"]:
        s1, s2 = _load_split_npz(os.path.join(out_dir, f"{split}.npz"))
        splits[split] = PatchDatasetS1S2(s1, s2)

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(splits["train"], batch_size, shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(splits["val"],   batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(splits["test"],  batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default="./runs/eps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_proj", default="s1_to_s2_ddpm_eps")
    # --- CodeCarbon ---
    parser.add_argument("--emissions", action="store_true",
                        help="Enable CodeCarbon emissions tracking")
    parser.add_argument("--cc_output_dir", default="./emissions_logs")
    parser.add_argument("--cc_measure_power_secs", type=int, default=15)
    parser.add_argument("--cc_tracking_mode", default="machine")
    parser.add_argument("--cc_gpu_ids", default="all")
    parser.add_argument("--cc_quiet", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(project=args.wandb_proj, config=vars(args))
    cc = EmissionsTrackerSession(
        enabled=args.emissions,
        project_name=(wandb.run.name if wandb.run else "train_eps"),
        output_dir=args.cc_output_dir,
        measure_power_secs=args.cc_measure_power_secs,
        tracking_mode=args.cc_tracking_mode,
        gpu_ids=args.cc_gpu_ids,
        quiet=True if args.cc_quiet else True,  # default quiet
    )
    cc.start()

    model = UNetCondMultiScale(
        in_ch=4,
        cond_ch=2,
        out_ch=4,
        base_ch=64,
        ch_mults=(1, 2, 4),
        t_dim=128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = DDPMSchedulerEPS(
        timesteps=args.timesteps,
        device=device
    )

    train_loader, val_loader, _ = load_dataloaders_from_npz(
        args.data_path, args.batch_size, args.num_workers
    )

    warmup_steps = len(train_loader) * 2
    lr_sched = LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[eps] Epoch {epoch}"):
                loss = loss_batch_eps(
                    model, batch, optimizer, device, scheduler
                )
                train_loss += float(loss)
                lr_sched.step()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss = loss_batch_eps(
                        model, batch, optimizer, device, scheduler
                    )
                    val_loss += float(loss)
            val_loss /= len(val_loader)

            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"[eps] Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}")
    finally:
        total_kg = cc.stop(wandb_run=wandb.run)
        if total_kg is not None:
            print(f"[Emissions] Total CO2eq: {total_kg:.6f} kg")

    print("Training complete.")


if __name__ == "__main__":
    main()
