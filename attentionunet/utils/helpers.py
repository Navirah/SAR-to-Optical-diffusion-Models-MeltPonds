from pathlib import Path
import os
import torch
from lightning.pytorch.loggers import WandbLogger
import json
from typing import Dict


def resolve_resume_path(user_arg: str | None, chkpt_dir: Path) -> str | None:
    """Resolve checkpoint path from user input or checkpoint directory."""
    if not user_arg:
        return None
    p = Path(user_arg)
    if p.is_absolute():
        return str(p)
    return str(chkpt_dir / p.name)


def set_env():
    """Set runtime environment flags for stability and performance."""
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64"
    )
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("medium")


def build_wandb_logger(project_name: str, run_name: str, output_dir):
    """Create a Weights & Biases logger."""
    return WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=str(output_dir),
    )


def build_denorm_fn_from_stats(stats_path: str, device: torch.device):
    """Build a denormalisation function from stored dataset statistics."""
    with open(stats_path, "r") as f:
        stats = json.load(f)

    def to_tensor(v):
        return torch.tensor(
            v, dtype=torch.float32, device=device
        ).view(1, -1, 1, 1)

    s2 = stats["s2"]
    if s2["method"] == "zscore":
        mean, std = to_tensor(s2["mean"]), to_tensor(s2["std"])
        denorm = lambda x: x * std + mean
    elif s2["method"] == "minmax":
        mn, mx = to_tensor(s2["min"]), to_tensor(s2["max"])
        denorm = lambda x: x * (mx - mn) + mn
    else:
        raise ValueError(f"Unknown method {s2['method']}")

    def fn(payload: Dict[str, torch.Tensor]):
        out = dict(payload)
        out["s2"] = denorm(out["s2"])
        out["pred"] = denorm(out["pred"])
        return out

    return fn
