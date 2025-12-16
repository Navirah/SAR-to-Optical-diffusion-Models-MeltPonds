from __future__ import annotations
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding followed by an MLP.
    """

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.dim = dim

        # Fixed random Fourier features
        self.register_buffer(
            "B",
            torch.randn(dim // 2) * scale,
            persistent=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) timesteps
        """
        t = t.float()
        angles = t[:, None] * self.B[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.mlp(emb)
