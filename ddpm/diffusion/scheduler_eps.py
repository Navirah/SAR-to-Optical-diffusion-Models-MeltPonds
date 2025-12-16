from __future__ import annotations
import torch
import math


class DDPMSchedulerEPS:
    """
    DDPM scheduler for ε-prediction (noise prediction).
    """

    def __init__(self, timesteps: int = 1000, s: float = 0.008, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        steps = torch.arange(timesteps + 1, dtype=torch.float64, device=device)

        def alpha_bar_fn(t):
            return torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2

        alpha_bars_full = alpha_bar_fn(steps)
        alpha_bars_full = alpha_bars_full / alpha_bars_full[0]

        alpha_bars = alpha_bars_full[1:]      # ᾱ₁ … ᾱ_T
        alpha_bars_prev = alpha_bars_full[:-1]

        betas = (1.0 - alpha_bars / alpha_bars_prev).clamp(0.0, 0.999)
        alphas = 1.0 - betas

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bars = alpha_bars.float()

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    #helpers
    def sample_timesteps(self, batch_size: int):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        """
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * noise

    # Sampling (ancestral DDPM)
    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        cond=None,
        cfg_w: float = 2.0,
        seed: int | None = None,
    ):
        """
        Standard DDPM sampling where model predicts ε.
        """
        B, C, H, W = shape
        device = self.device

        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            randn = lambda *s: torch.randn(*s, device=device, generator=g)
        else:
            randn = lambda *s: torch.randn(*s, device=device)

        x = randn(B, C, H, W)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            eps_cond = model(x, t_tensor, cond)

            if cfg_w is not None:
                eps_uncond = model(x, t_tensor, torch.zeros_like(cond))
                eps = eps_uncond + cfg_w * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            mean = coef1 * (x - coef2 * eps)

            if t > 0:
                noise = randn(B, C, H, W)
                x = mean + torch.sqrt(self.betas[t]) * noise
            else:
                x = mean

        return x
