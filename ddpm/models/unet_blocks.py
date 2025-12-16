from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two Conv2d + GroupNorm + SiLU blocks, with optional timestep projection add.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int | None = None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.t_proj = nn.Linear(t_dim, out_ch) if t_dim is not None else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.block(x)
        if self.t_proj is not None and t_emb is not None:
            h = h + self.t_proj(t_emb)[:, :, None, None]
        return h


class DownBlock(nn.Module):
    """
    DoubleConv then strided conv (3x3, stride=2) for downsampling.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, t_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x, t_emb)
        return self.pool(h), h  # downsampled, skip


class UpBlock(nn.Module):
    """
    Transposed-conv upsample then concat skip then DoubleConv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, t_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch (odd dims)
        if x.shape[2:] != skip.shape[2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t_emb)
