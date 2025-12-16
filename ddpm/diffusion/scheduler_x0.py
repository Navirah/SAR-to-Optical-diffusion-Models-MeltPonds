from __future__ import annotations
import torch
import math


class DDPMX0Scheduler:
    """
    DDPM scheduler for x₀-prediction (clean image prediction).
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine", device="cpu"):
        self.timesteps = num_timesteps
        self.device = device

        if schedule != "cosine":
            raise ValueError("Only cosine schedule is supported.")

        steps = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
        s = 0.008

        def alpha_bar_fn(t):
            return torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2

        alpha_bars_full = alpha_bar_fn(steps)
        alpha_bars_full = alpha_bars_full / alpha_bars_full[0]

        self.alpha_bars = alpha_bars_full[1:].float()
        self.alpha_bars_prev = alpha_bars_full[:-1].float()

        self.betas = (1.0 - self.alpha_bars / self.alpha_bars_prev).clamp(0.0, 0.999)
        self.alphas = 1.0 - self.betas

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    #helpers
    def sample_timesteps(self, batch_size: int):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * noise

    #sampling
    @torch.no_grad()
    def sample(
        self,
        model,
        cond,
        shape,
        steps: int | None = None,
        seed: int | None = None,
        progress: bool = False,
    ):
        """
        DDPM sampling when model predicts x₀ directly.
        """
        B, C, H, W = shape
        device = self.device

        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            randn = lambda *s: torch.randn(*s, device=device, generator=g)
        else:
            randn = lambda *s: torch.randn(*s, device=device)

        x = randn(B, C, H, W)

        T = self.timesteps
        ts = range(T - 1, -1, -1)

        for t in ts:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            x0_pred = model(x, t_tensor, cond)

            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars_prev[t]

            coef1 = torch.sqrt(alpha_bar_prev)
            coef2 = torch.sqrt(1.0 - alpha_bar_prev)

            mean = coef1 * x0_pred + coef2 * (x - torch.sqrt(alpha_bar) * x0_pred) / torch.sqrt(1.0 - alpha_bar)

            if t > 0:
                x = mean + torch.sqrt(self.betas[t]) * randn(B, C, H, W)
            else:
                x = mean

        return x
