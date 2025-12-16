import argparse
from pathlib import Path
import torch
import wandb

from lightning.pytorch.loggers import WandbLogger

from configs.config import get_config
from lightning import utils as lightning_utils

from utils.helpers import (
    set_env,
    resolve_resume_path,
    build_wandb_logger,
)

from lightning.callbacks.factory import build_callbacks


def main(
    config_path: str,
    checkpoint_path: str | None = None,
    run_name_override: str | None = None,
):
    # runtime and torch environment setup
    set_env()

    #config
    config = get_config()

    #data
    data_path = Path(config.data.dataset_path)
    config = lightning_utils.configure_location_args(config, data_path)

    train_loader, val_loader = lightning_utils.build_dataloaders(
        config, lambda x: x, num_workers=8
    )

    #model
    model = lightning_utils.build_model(config)

    #outputs
    base_output_dir = Path(
        getattr(
            config,
            "base_output_dir",
            "/home/nkamal/projects/colocation/diffusion_model_alex/output_emissions",
        )
    )
    project_name = getattr(config, "project_name", "default_project")
    run_name = run_name_override or getattr(config, "run_name", "run")

    output_dir = base_output_dir / project_name / run_name
    chkpt_dir = output_dir / "chkpts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(config, "training"):
        raise RuntimeError("config.training missing")

    # Lightning uses this as the default checkpoint root
    setattr(config.training, "default_root_dir", str(output_dir))

    #logging
    logger = build_wandb_logger(project_name, run_name, output_dir)

    #callbacks
    callbacks, callback_config = build_callbacks(
        config=config,
        output_dir=output_dir,
        chkpt_dir=chkpt_dir,
        project_name=project_name,
        run_name=run_name,
    )

    #trainer
    grad_clip = getattr(config.optim, "grad_clip", 0.0)
    trainer = lightning_utils.build_trainer(
        config.training,
        grad_clip,
        callback_config,
        config.precision,
        getattr(config, "device", None),
        logger,
    )

    #resume
    resume_ckpt = resolve_resume_path(checkpoint_path, chkpt_dir)
    if resume_ckpt and not Path(resume_ckpt).exists():
        print(f"[warn] Checkpoint not found: {resume_ckpt}")
        resume_ckpt = None

    print(f"Output dir    : {output_dir}")
    print(f"Checkpoint dir: {chkpt_dir}")
    print(f"Resume ckpt   : {resume_ckpt or 'None'}")

    #train
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    final_path = chkpt_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_path))
    print(f"[ok] Saved final checkpoint -> {final_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, default=None)
    parser.add_argument("-C", "--checkpoint", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    main(args.config_path, args.checkpoint, args.run_name)
