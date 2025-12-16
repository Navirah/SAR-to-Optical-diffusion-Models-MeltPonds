"""Credits: Alex Soulis, UCL"""
import torch
import torch.nn as nn

from ...sampling.k_sampling import append_dims


class AbstractKarrasDenoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, device="cpu"):
        super().__init__()
        self.inner_model = inner_model

        # Keep buffers on CPU; don't persist in checkpoints.
        self.register_buffer("ignore", torch.tensor([1.0], dtype=torch.float32), persistent=False)

        self.trim_output_fields = lambda x: x
        self.sigma_converter = lambda x: x  # identity by default

    def set_buffer_width(self, pixel_width):
        self.trim_output_fields = lambda x: x[
            :, :, slice(pixel_width, -pixel_width), slice(pixel_width, -pixel_width)
        ]

    # ---------- weighting helpers (safe: cast to sigma device/dtype on use) ----------
    def _weighting_soft_min_snr(self, sigma):
        sd = self.sigma_data.to(device=sigma.device, dtype=sigma.dtype)
        return (sigma * sd) ** 2 / (sigma**2 + sd**2) ** 2

    def _weighting_snr(self, sigma):
        sd = self.sigma_data.to(device=sigma.device, dtype=sigma.dtype)
        return sd**2 / (sigma**2 + sd**2)

    # @torch.compile
    def loss(self, input, condition, **kwargs):
        """
        Training loss per output-channel. We do not coerce dtype/device here to keep
        training numerics unchanged; only use-time casts inside helpers are applied.
        """
        sigma = self.sample_training_sigmas(input)

        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)

        noised_input = input + torch.randn_like(input) * append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, condition, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out

        img_flattened = (
            (self.trim_output_fields(model_output) - self.trim_output_fields(target)) ** 2
        ).flatten(2).mean(2)

        return (img_flattened * c_weight.unsqueeze(1)).mean(dim=0)

    # @torch.compile
    def forward(self, input, cond, sigma, **kwargs):
        """
        Inference-time preconditioned forward.
        Here we coerce inputs to the inner model's param dtype/device to avoid mismatches.
        """
        p0 = next(self.inner_model.parameters())
        model_dev, model_dtype = p0.device, p0.dtype

        if not self.training:
            input = input.to(model_dev, model_dtype)
            if cond is not None:
                if isinstance(cond, (tuple, list)):
                    cond = tuple(c.to(model_dev, model_dtype) if torch.is_tensor(c) else c for c in cond)
                elif torch.is_tensor(cond):
                    cond = cond.to(model_dev, model_dtype)
            sigma = sigma.to(model_dev, model_dtype)

        # get scalings (subclasses cast to sigma's device/dtype internally)
        c_skip, c_out, c_in = self.get_scalings(sigma)

        if not self.training:
            c_skip, c_out, c_in = [x.to(model_dev, model_dtype) for x in (c_skip, c_out, c_in)]
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in (c_skip, c_out, c_in)]

        c_noise = self.sigma_converter(sigma)
        if not self.training and torch.is_tensor(c_noise):
            c_noise = c_noise.to(model_dev, model_dtype)

        return self.inner_model(input * c_in, cond, c_noise, **kwargs) * c_out + input * c_skip


class EDMDenoiser(AbstractKarrasDenoiser):
    """EDM preconditioning and training schedule."""

    def __init__(self, inner_model, sigma_data=1.0, weighting="karras", device="cpu"):
        super().__init__(inner_model, device)
        self.inner_model = inner_model

        # Keep constants as non-persistent CPU buffers; cast at use-time.
        self.register_buffer("sigma_data", torch.tensor([sigma_data], dtype=torch.float32), persistent=False)
        self.get_scalings = self.get_edm_scalings

        # Log-normal training distribution params
        self.register_buffer("P_std",  torch.tensor([1.2],  dtype=torch.float32), persistent=False)
        self.register_buffer("P_mean", torch.tensor([-1.2], dtype=torch.float32), persistent=False)

        # Sigma range
        self.register_buffer("sigma_max", torch.tensor([80.0],  dtype=torch.float32), persistent=False)
        self.register_buffer("sigma_min", torch.tensor([0.002], dtype=torch.float32), persistent=False)

        if callable(weighting):
            self.weighting = weighting
        elif weighting == "karras":
            self.weighting = torch.ones_like
        elif weighting == "soft-min-snr":
            self.weighting = self._weighting_soft_min_snr
        elif weighting == "snr":
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f"Unknown weighting type {weighting}")

    def get_edm_scalings(self, sigma):
        """EDM scalings; cast to sigma's device/dtype on use."""
        sd = self.sigma_data.to(device=sigma.device, dtype=sigma.dtype)
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out  = sigma * sd / (sigma**2 + sd**2).sqrt()
        c_in   = 1.0 / (sigma**2 + sd**2).sqrt()
        return c_skip, c_out, c_in

    def sample_training_sigmas(self, input):
        """Sample noise levels during training using log-normal distribution."""
        batch_size = input.shape[0]
        device = input.device
        dtype  = input.dtype

        P_std  = self.P_std.to(device=device, dtype=dtype)
        P_mean = self.P_mean.to(device=device, dtype=dtype)

        log_sigma = torch.randn(batch_size, device=device, dtype=dtype) * P_std + P_mean
        sigma = torch.exp(log_sigma)

        # Clamp on host values; OK since .item() is scalar
        sigma = torch.clamp(sigma, self.sigma_min.item(), self.sigma_max.item())
        return sigma


class VPDenoiser(AbstractKarrasDenoiser):
    """Variance-Preserving SDE in the EDM formulation (for comparison)."""

    def __init__(self, inner_model, device="cpu"):
        super().__init__(inner_model, device)
        self.inner_model = inner_model
        self.get_scalings = self.get_vpsde_scalings

        # Keep constants as non-persistent CPU buffers; cast at use-time.
        self.register_buffer("eps_t",     torch.tensor([1e-4],   dtype=torch.float32), persistent=False)
        self.register_buffer("M",         torch.tensor([1000.0], dtype=torch.float32), persistent=False)
        self.register_buffer("uniform_u", torch.tensor([1.0],    dtype=torch.float32), persistent=False)
        self.uniform_l = self.eps_t  # already a buffer

        # Local constants used in converter lambdas (CPU tensors are fine; math casts at call sites)
        beta_min = torch.tensor([0.1],  dtype=torch.float32)
        beta_d   = torch.tensor([19.9], dtype=torch.float32)

        self.convert_sigma_to_t = (
            lambda sigma: (
                ((beta_min**2 + 2 * beta_d * (1 + sigma**2).log())).sqrt() - beta_min
            ) / beta_d
        )
        self.convert_t_to_sigma = (
            lambda t: ((0.5 * beta_d * t**2 + beta_min * t).exp() - 1).sqrt()
        )

        self.weighting = torch.ones_like

    def get_vpsde_scalings(self, sigma):
        # Keep everything on sigma's device/dtype
        c_skip = self.ignore.to(device=sigma.device, dtype=sigma.dtype)
        c_out  = -sigma
        one    = torch.ones_like(sigma)
        c_in   = 1.0 / (sigma**2 + one).sqrt()
        return c_skip, c_out, c_in

    def sample_training_sigmas(self, input):
        device = input.device
        dtype  = input.dtype
        u = self.uniform_u.to(device=device, dtype=dtype)
        l = self.uniform_l.to(device=device, dtype=dtype)
        t = (u - l) * torch.rand([input.shape[0]], device=device, dtype=dtype) + l
        return self.convert_t_to_sigma(t)
