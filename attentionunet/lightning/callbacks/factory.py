from pathlib import Path

from adapter import KarrasSchedulerAdapter
from lightning.callbacks import DiffusionDebugCallback, CodeCarbonCallback


def build_callbacks(
    config,
    output_dir: Path,
    chkpt_dir: Path,
    project_name: str,
    run_name: str,
):
    """
    Build Lightning callbacks and callback_config dict.

    returns - 
    callbacks : list
        List of instantiated Lightning callbacks.
    callback_config : dict
        Configuration dict expected by lightning_utils.build_trainer().
    """

    #base callback config
    callback_config = {
        "checkpoint_dir": str(chkpt_dir),
        "lr_monitor": "step",
        "ema_rate": getattr(config.model, "ema_rate", None),
        "save_n_epochs": getattr(config.training, "save_n_epochs", None),
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 10,
    }

    #visualization callback
    viz_scheduler = KarrasSchedulerAdapter(
        n=20,
        rho=7,
        sigma_min=0.002,
        sigma_max=80.0,
        sampler="heun",
    )

    viz_cb = DiffusionDebugCallback(
        scheduler=viz_scheduler,
        every_n_epochs=1,
        save_dir=str(output_dir / "previews"),
        n_samples=8,
        steps=20,
        seed=1234,
        cfg_w=None,
        predicts="x0",
    )

    #emissions tracking
    emissions_dir = output_dir / "emissions_logs"
    emissions_dir.mkdir(parents=True, exist_ok=True)

    emissions_cb = CodeCarbonCallback(
        project_name=f"{project_name}/{run_name}",
        output_dir=str(emissions_dir),
        measure_power_secs=60,
        tracking_mode="machine",
        gpu_ids="all",
        quiet=True,
    )

    # Final assembly
    callbacks = [viz_cb, emissions_cb]
    callback_config["custom_callbacks"] = callbacks

    return callbacks, callback_config
