from typing import Tuple, Sequence
import torch
from torch import Tensor


def _as_tensor(x, device) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    raise TypeError(f"Expected Tensor, got {type(x)}")


def extract_s1_s2(batch, device) -> Tuple[Tensor, Tensor]:
    if isinstance(batch, dict):
        for k1, k2 in (("s1", "s2"), ("input", "target"), ("x", "y")):
            if k1 in batch and k2 in batch:
                return _as_tensor(batch[k1], device), _as_tensor(batch[k2], device)

    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return extract_s1_s2(batch[0], device)
        if isinstance(batch[0], dict):
            return extract_s1_s2(batch[0], device)
        if len(batch) >= 2:
            return _as_tensor(batch[0], device), _as_tensor(batch[1], device)

    raise TypeError(f"Unrecognized batch structure: {type(batch)}")


def cat_horiz(imgs: Sequence[Tensor]) -> Tensor:
    b, c, h, w = imgs[0].shape
    for t in imgs:
        assert t.shape == (b, c, h, w)
    return torch.cat(imgs, dim=3)
