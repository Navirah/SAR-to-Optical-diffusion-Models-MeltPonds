from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.time_embedding import TimeEmbedding
from models.unet_blocks import DoubleConv, DownBlock, UpBlock


class UNetCondMultiScale(nn.Module):
    """
    Conditional U-Net with multi-scale S1 conditioning.
    Inputs:
      x_t  : (B, 4, H, W)  noised S2
      t    : (B,)         diffusion timestep
      cond : (B, 2, H, W) S1 (HH, HV)

    Output:
      (B, 4, H, W) prediction (ε or x0 depending on training)
    """

    def __init__(
        self,
        in_ch: int = 4,
        cond_ch: int = 2,
        out_ch: int = 4,
        base_ch: int = 32,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        t_dim: int = 128,
        out_activation: str | None = None,
    ):
        super().__init__()
        self.t_emb = TimeEmbedding(t_dim)

        # ---- multi-scale S1 projections ----
        self.cond_proj0 = nn.Conv2d(cond_ch, base_ch * ch_mults[0], 1)
        self.cond_proj1 = nn.Conv2d(cond_ch, base_ch * ch_mults[1], 1)
        self.cond_proj2 = nn.Conv2d(cond_ch, base_ch * ch_mults[2], 1)

        # ---- encoder ----
        self.in_conv = DoubleConv(
            in_ch + cond_ch, base_ch * ch_mults[0], t_dim
        )

        self.down1 = DownBlock(
            base_ch * ch_mults[0] + base_ch * ch_mults[0],
            base_ch * ch_mults[1],
            t_dim,
        )

        self.down2 = DownBlock(
            base_ch * ch_mults[1] + base_ch * ch_mults[1],
            base_ch * ch_mults[2],
            t_dim,
        )

        # ---- bottleneck ----
        self.mid = DoubleConv(
            base_ch * ch_mults[2] + base_ch * ch_mults[2],
            base_ch * ch_mults[2],
            t_dim,
        )

        # ---- decoder ----
        self.up2 = UpBlock(
            base_ch * ch_mults[2],
            base_ch * ch_mults[2],
            base_ch * ch_mults[1],
            t_dim,
        )

        self.up1 = UpBlock(
            base_ch * ch_mults[1],
            base_ch * ch_mults[1],
            base_ch * ch_mults[0],
            t_dim,
        )

        # ---- output head ----
        last = nn.Conv2d(base_ch * ch_mults[0], out_ch, kernel_size=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        self.out = nn.Sequential(
            nn.Conv2d(base_ch * ch_mults[0], base_ch * ch_mults[0], 3, padding=1),
            nn.SiLU(),
            last,
        )

        if out_activation is None:
            self._act = nn.Identity()
        elif out_activation.lower() == "tanh":
            self._act = nn.Tanh()
        elif out_activation.lower() == "sigmoid":
            self._act = nn.Sigmoid()
        else:
            raise ValueError("out_activation must be one of {None, 'tanh', 'sigmoid'}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        :param x_t:  noised S2 (B, 4, H, W)
        :param t:    timesteps (B,)
        :param cond: S1 conditioning (B, 2, H, W)
        """
        t_emb = self.t_emb(t)

        # ---- condition projections ----
        cond0 = self.cond_proj0(cond)
        cond1 = self.cond_proj1(F.avg_pool2d(cond, 2))
        cond2 = self.cond_proj2(F.avg_pool2d(cond, 4))

        # ---- encoder ----
        h0 = self.in_conv(torch.cat([x_t, cond], dim=1), t_emb)

        d1, s1 = self.down1(torch.cat([h0, cond0], dim=1), t_emb)
        d2, s2 = self.down2(torch.cat([d1, cond1], dim=1), t_emb)

        # ---- bottleneck ----
        m = self.mid(torch.cat([d2, cond2], dim=1), t_emb)

        # ---- decoder ----
        u2 = self.up2(m, s2, t_emb)
        u1 = self.up1(u2, s1, t_emb)

        return self._act(self.out(u1))
