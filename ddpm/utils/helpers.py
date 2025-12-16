import torch
import numpy as np
import torchvision.utils as tv


#S2 visualization helpers
def s2_to_rgb01_z(s2: torch.Tensor) -> torch.Tensor:
    """
    Convert z-scored Sentinel-2 tensor to RGB in [0,1].

    Assumes channel order:
      S2 = (B2, B3, B4, B8)  → use RGB = (B4, B3, B2)

    :param s2: Tensor (B, 4, H, W)
    :return: Tensor (B, 3, H, W) clipped to [0,1]
    """
    assert s2.ndim == 4 and s2.size(1) >= 3

    # RGB = (B4, B3, B2)
    rgb = torch.stack(
        [s2[:, 2], s2[:, 1], s2[:, 0]],
        dim=1
    )

    # normalize per-image for visualization
    b, c, h, w = rgb.shape
    rgb = rgb.view(b, c, -1)
    minv = rgb.min(dim=-1, keepdim=True)[0]
    maxv = rgb.max(dim=-1, keepdim=True)[0]
    rgb = (rgb - minv) / (maxv - minv + 1e-8)
    rgb = rgb.view(b, c, h, w)

    return rgb.clamp(0.0, 1.0)


#S1 visualization helpers
def s1_gray_vis(s1: torch.Tensor) -> torch.Tensor:
    """
    Convert Sentinel-1 (HH, HV) to a single grayscale image for visualization.

    Uses HH band by default.

    :param s1: Tensor (B, 2, H, W)
    :return: Tensor (B, 3, H, W) for grid display
    """
    assert s1.ndim == 4 and s1.size(1) >= 1

    hh = s1[:, 0:1]  # (B,1,H,W)

    # min-max normalize per image
    b, _, h, w = hh.shape
    x = hh.view(b, -1)
    minv = x.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    maxv = x.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    hh = (hh - minv) / (maxv - minv + 1e-8)

    # repeat to 3 channels for torchvision grids
    return hh.repeat(1, 3, 1, 1).clamp(0.0, 1.0)


#Grid helpers
def make_triptych_grid(
    s1_vis: torch.Tensor,
    s2_gt: torch.Tensor,
    s2_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Create an interleaved grid:
      [S1 | S2 GT | S2 Pred] repeated for each sample.

    :param s1_vis:  (B, 3, H, W)
    :param s2_gt:   (B, 3, H, W)
    :param s2_pred: (B, 3, H, W)
    :return: grid tensor suitable for save_image
    """
    assert s1_vis.shape == s2_gt.shape == s2_pred.shape

    panels = []
    for i in range(s1_vis.size(0)):
        panels.extend([s1_vis[i], s2_gt[i], s2_pred[i]])

    grid = tv.make_grid(
        panels,
        nrow=3,
        padding=2
    )
    return grid
