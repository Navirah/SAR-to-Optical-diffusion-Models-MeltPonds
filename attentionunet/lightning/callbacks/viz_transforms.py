import torch
from torch import Tensor

from .viz_stats import S1_MEAN, S1_STD, S2_MEAN, S2_STD, S2_TONEMAP_DIV


def _unz(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    mean = mean.to(x.device).view(1, -1, 1, 1)
    std  = std.to(x.device).view(1, -1, 1, 1)
    return x * std + mean


def s1_gray_vis(s1_norm: Tensor) -> Tensor:
    phys = _unz(s1_norm, S1_MEAN, S1_STD)
    vv = phys[:, 0:1]
    vmin = vv.amin(dim=(2, 3), keepdim=True)
    vmax = vv.amax(dim=(2, 3), keepdim=True)
    return (vv - vmin) / (vmax - vmin + 1e-8)


def s2_rgb_from_phys01(s2_phys: Tensor) -> Tensor:
    r = s2_phys[:, 2:3]
    g = s2_phys[:, 1:2]
    b = s2_phys[:, 0:1]
    rgb = torch.cat([r, g, b], dim=1)
    rgb = rgb / S2_TONEMAP_DIV
    return rgb.clamp(0, 1)


def s2_to_rgb01_from_norm(s2_norm: Tensor) -> Tensor:
    phys = _unz(s2_norm, S2_MEAN, S2_STD)
    return s2_rgb_from_phys01(phys)
