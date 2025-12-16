"""Credit of parts of code to Alex Soulis, UCL"""

from pathlib import Path
import torch
import xarray as xr
import numpy as np
import lightning as pl

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks import Callback as _PLCallback

#data utilities

from ..data.scaling import DataScaler
from ..data.data_loading import get_dataloader, prepare_and_scale_data
from ..data.constants import TRAINING_COORDS_LOOKUP
from ..data.reader import (
    setup_experiment_compat,
    setup_dataset,
    setup_dataset_test,
)

#backbone (architecture only)

from ..yang_cm.utils import create_model as create_cm_model

#diffusion lightning module and setup

from models.diffusion import (
    LightningDiffusion,
    setup_edm_model,
    setup_vp_model,
)

#ema support (optional)

try:
    from .ema import EMA as _RawEMA
except Exception:
    _RawEMA = None


#model construction

def build_model(config, checkpoint_name: str | None = None):
    """
    Build a diffusion-only Lightning model.

    This function:
    - constructs the base UNet backbone
    - wraps it with the chosen diffusion formulation
    - optionally restores weights from a checkpoint
    """
    # backbone network
    base_model = create_cm_model(config)

    # diffusion formulation
    diffusion_type = config.diffusion_type

    if diffusion_type == "karras":
        base_model, loss_config = setup_edm_model(
            config, base_model, config.device
        )
    elif diffusion_type == "vpsde":
        base_model, loss_config = setup_vp_model(
            config, base_model, config.device
        )
    else:
        raise ValueError(f"Unknown diffusion_type: {diffusion_type}")

    # lightning wrapper configuration
    model_kwargs = {
        "output_channels": config.data.variables[1],
        "weights": config.training.loss_weights,
    }

    if checkpoint_name is None:
        model = LightningDiffusion(
            model=base_model,
            loss_config=loss_config,
            optimizer_config=config.optim,
            **model_kwargs,
        )
    else:
        if not Path(checkpoint_name).is_file():
            base_output_dir = Path(config.base_output_dir)
            output_dir = base_output_dir / config.project_name / config.run_name
            checkpoint_name = output_dir / "chkpts" / checkpoint_name

        model = LightningDiffusion.load_from_checkpoint(
            checkpoint_path=str(checkpoint_name),
            model=base_model,
            loss_config=loss_config,
            optimizer_config=config.optim,
            **model_kwargs,
        )

    return model


#trainer utilities

def convert_precision(precision_string: str) -> str:
    """Map config precision strings to Lightning-compatible values."""
    conversion = {
        "32": "32",
        "16": "16-mixed",
        "bf16": "bf16-mixed",
    }
    return conversion[precision_string]


def build_trainer(
    training_config,
    gradient_clip_val: float,
    callback_config: dict,
    precision: str,
    device,
    logger,
):
    callbacks = get_callbacks(callback_config)

    # optional visibility into callback stack
    for i, cb in enumerate(callbacks):
        print(f"[callbacks] idx={i} type={type(cb)}")

    trainer_args = get_training_config(
        training_config, gradient_clip_val, device
    )
    trainer_args["callbacks"] = callbacks

    trainer = pl.Trainer(
        precision=convert_precision(precision),
        devices=1,
        accumulate_grad_batches=16,
        logger=logger,
        **trainer_args,
    )
    return trainer


def get_training_config(training_config, gradient_clip_val, device):
    """Extract core trainer arguments from training config."""
    return {
        "accelerator": "gpu",
        "max_epochs": training_config.n_epochs,
        "gradient_clip_val": gradient_clip_val,
    }


#callbacks

def get_callbacks(callback_args: dict):
    args = dict(callback_args)
    callbacks = []

    # exponential moving average (if available)
    ema_rate = args.get("ema_rate")
    if ema_rate is not None and _RawEMA is not None:
        try:
            if isinstance(_RawEMA, type) and issubclass(_RawEMA, _PLCallback):
                callbacks.append(_RawEMA(ema_rate))
        except Exception as e:
            print(f"[callbacks] EMA skipped: {e}")

    # learning rate logging
    lr_monitor_interval = args.get("lr_monitor")
    if lr_monitor_interval is not None:
        callbacks.append(
            LearningRateMonitor(logging_interval=lr_monitor_interval)
        )

    # checkpointing
    checkpoint_dir = args.pop("checkpoint_dir")
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}",
            every_n_epochs=1,
            save_top_k=10,
            monitor="val_loss",
        )
    )

    save_n_epochs = args.get("save_n_epochs")
    if save_n_epochs is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{val_loss:.4f}",
                every_n_epochs=save_n_epochs,
                save_top_k=-1,
                monitor=None,
            )
        )

    # custom callbacks (visualisation, emissions, etc.)
    custom = args.get("custom_callbacks")
    if custom:
        callbacks.extend(custom if isinstance(custom, (list, tuple)) else [custom])

    # ensure Lightning-compatible callbacks only
    callbacks = [
        cb for cb in callbacks
        if isinstance(cb, _PLCallback) and cb is not None
    ]

    return callbacks


#data scalers

def build_or_load_data_scaler(config, data_scaler_parameters_path=None):
    """Create a new scaler or load one from disk."""
    if data_scaler_parameters_path is None:
        data_scaler = build_data_scaler(
            Path(config.data.dataset_path),
            config.data.variable_scaler_map,
            config.data.location_config,
            config.data.train_indices,
        )
    else:
        data_scaler = load_data_scaler(data_scaler_parameters_path)
    return data_scaler


def build_data_scaler(
    data_path,
    variable_scaler_map,
    variable_location_config,
    split_config,
):
    split = create_indices(split_config)
    ds = prepare_and_scale_data(
        data_path,
        split,
        variable_location_config,
        data_transform=None,
    )
    scaler = DataScaler(variable_scaler_map)
    scaler.fit(ds)
    ds.close()
    return scaler


def load_data_scaler(parameters_path):
    scaler = DataScaler({})
    scaler.load_scaler_parameters(parameters_path)
    return scaler


#location conditioning

def configure_location_args(config, data_path):
    """Attach latitude/longitude grids for learnable location parameters."""
    if config.model.location_parameters is None:
        config.model.location_parameter_config = None
        return config

    with xr.open_dataset(data_path) as ds:
        coords = ds.lat.values, ds.lon.values

    config.model.location_parameter_config = (
        coords,
        config.model.location_parameters,
    )
    return config


def setup_custom_training_coords(config, sampling_config):
    """Set training coordinates for region-specific sampling."""
    if config.model.location_parameters is None:
        config.data.training_coords = None
    else:
        config.data.training_coords = TRAINING_COORDS_LOOKUP[
            sampling_config.training_dataset
        ]
    return config


#indices and splits

def create_custom_indices(indices_config: dict):
    indices = []
    for indices_type, cfg in indices_config.items():
        if indices_type == "date_range":
            start, end = cfg
            new_indices = xr.date_range(
                np.datetime64(start), np.datetime64(end)
            )
        elif indices_type == "isel":
            new_indices = cfg
        else:
            continue
        indices.append(new_indices)
    return np.concatenate(indices) if indices else np.array([])


def create_indices(full_indices_config):
    cfg = dict(full_indices_config)
    split = np.array(cfg.pop("split"))
    if cfg:
        return split, create_custom_indices(cfg)
    return split


#dataloaders

def build_dataloaders(config, transform, num_workers):
    """Create training and validation dataloaders."""
    if getattr(config, "tfrecord", None):
        exp_config = setup_experiment_compat(config)
        exp_config.BATCH_SIZE = config.training.batch_size
        exp_config.SHUFFLE = True
        exp_config.NUM_WORKERS = num_workers
        train_loader, val_loader = setup_dataset(exp_config)
        return train_loader, val_loader

    dl_cfgs = [
        (config.data.train_indices, True, False),
        (config.data.eval_indices, False, True),
    ]

    loaders = (
        build_dataloader(
            config.data.dataset_path,
            config.data.variables,
            indices,
            transform,
            config.data.variable_location,
            config.data.location_config,
            config.data.image_size,
            config.training.loss_buffer_width,
            config.data.training_coords,
            config.training.batch_size,
            shuffle=shuffle,
            evaluation=evaluation,
            num_workers=num_workers,
        )
        for indices, shuffle, evaluation in dl_cfgs
    )
    return loaders


def build_dataloaders_test(config, transform, num_workers):
    """Create test-only dataloader for inference."""
    if getattr(config, "tfrecord", None):
        exp_config = setup_experiment_compat(config)
        exp_config.BATCH_SIZE = config.training.batch_size
        exp_config.SHUFFLE = False
        exp_config.NUM_WORKERS = num_workers
        return setup_dataset_test(exp_config)
    raise NotImplementedError


def build_dataloader(
    data_path,
    variables,
    indices,
    transform,
    variable_location,
    location_config,
    image_size,
    buffer_width,
    training_coords,
    batch_size,
    shuffle,
    evaluation,
    num_workers,
):
    formatted_indices = create_indices(indices)
    return get_dataloader(
        data_path,
        variables,
        transform,
        include_time_inputs=False,
        variable_location=variable_location,
        location_config=location_config,
        image_size=image_size,
        buffer_width=buffer_width,
        training_coords=training_coords,
        batch_size=batch_size,
        split=formatted_indices,
        evaluation=evaluation,
        shuffle=shuffle,
        num_workers=num_workers,
    )
