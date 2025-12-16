"""An assortment of schedule and sampling functions for diffusion models.

These have all been directly taken and sometimes adapted from k-diffusion:
https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py

Minor adaptations have been made to incorporate conditional information,
as well as earlier SDEs (VP) in the build_to_d_vp and edm_type parameters. 
"""

import math
import torch
from tqdm.auto import trange


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_karras_sqrt(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.sqrt(sigmas)
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(
        ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)
    )
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def get_sigmas_ve(n, sigma_min=0.002, sigma_max=80, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    ramp = torch.linspace(0, 1, n)
    sigmas = sigma_max**2 * (sigma_min**2 / sigma_max**2) ** ramp
    return append_zero(sigmas).to(device)


def get_sigmas_subvp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    t_to_sigma = lambda t: 1 - (-(0.5 * beta_d * (t**2) + beta_min * t)).exp()
    sigmas = t_to_sigma(t)
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_d_sqrt(x, sigma, denoised):
    sigma_deriv_ratio = 0.5 * 1 / sigma
    return (x - denoised) * append_dims(sigma_deriv_ratio, x.ndim)


def build_to_d_vp(beta_min, beta_d, device="cuda"):
    """Assumed fixed constants for simplicity."""
    beta_min = torch.Tensor([beta_min]).to(device)
    beta_d = torch.Tensor([beta_d]).to(device)
    t_to_sigma = lambda t: (torch.exp(0.5 * beta_d * (t**2) + beta_min * t) - 1).sqrt()
    sigma_deriv = (
        lambda t: 0.5 * (beta_min + beta_d * t) * (t_to_sigma(t) + 1 / t_to_sigma(t))
    )
    sigma_inv = (
        lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    s = lambda t: 1 / (1 + t_to_sigma(t) ** 2).sqrt()
    s_deriv = lambda t: -t_to_sigma(t) * sigma_deriv(t) * (s(t) ** 3)

    def to_d_vp(x, sigma, denoised):
        t = sigma_inv(sigma)
        return (sigma_deriv(t) / t_to_sigma(t) + s_deriv(t) / s(t)) * x - sigma_deriv(
            t
        ) * s(t) / t_to_sigma(t) * denoised

    def sigma_to_s(sigma):
        t = sigma_inv(sigma)
        return s(t)

    return to_d_vp, sigma_to_s, sigma_inv


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_to_d(sde_type):
    """Construct derivative required to integrate SDE in the 
    Karras 2022 formalism. Simple for 'edm', but scaling and 
    noise schedule in the variance preserving (vp) integration requires some
    fiddling. 
    """
    const_scaling = lambda sigma: torch.ones_like(sigma, device=sigma.device)
    identity = lambda sigma: sigma
    if sde_type == "edm":
        return to_d, const_scaling, identity
    elif sde_type == "sqrt":
        return to_d_sqrt, const_scaling, identity
    elif sde_type == "vp":
        return build_to_d_vp(0.1, 19.0, device="cuda")


@torch.no_grad()
def sample_euler(
    model,
    x,
    cond,
    sigmas,
    sde_type="edm",
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    to_d, s, sigma_inv = get_to_d(sde_type)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = (
                s(sigma_hat) / s(sigmas[i]) * x
                + s(sigma_hat)
                * eps
                * (sigma_hat**2 - sigmas[i] ** 2).clip(min=0) ** 0.5
            )
        denoised = model(x / s(sigma_hat), cond, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigma_inv(sigmas[i + 1]) - sigma_inv(sigma_hat)
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


# @torch.no_grad()
# def sample_heun(
#     model,
#     x,
#     cond,
#     sigmas,
#     t_converter=None,
#     extra_args=None,
#     callback=None,
#     disable=None,
#     s_churn=0.0,
#     s_tmin=0.0,
#     s_tmax=float("inf"),
#     s_noise=1.0,
# ):
#     """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
#     model.eval()
#     extra_args = (
#         {"sigma_to_t_callable": t_converter} if extra_args is None else extra_args
#     )
#     s_in = x.new_ones([x.shape[0]])
#     for i in trange(len(sigmas) - 1, disable=disable):
#         gamma = (
#             min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#             if s_tmin <= sigmas[i] <= s_tmax
#             else 0.0
#         )
#         eps = torch.randn_like(x) * s_noise
#         sigma_hat = sigmas[i] * (gamma + 1)
#         if gamma > 0:
#             x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
#         if t_converter is not None:
#             denoised = model(x, cond, sigma_hat * s_in)
#         else:
#             denoised = model(x, cond, sigma_hat * s_in)
#         d = to_d(x, sigma_hat, denoised)
#         if callback is not None:
#             callback(
#                 {
#                     "x": x,
#                     "i": i,
#                     "sigma": sigmas[i],
#                     "sigma_hat": sigma_hat,
#                     "denoised": denoised,
#                 }
#             )
#         dt = sigmas[i + 1] - sigma_hat
#         if sigmas[i + 1] == 0:
#             # Euler method
#             x = x + d * dt
#         else:
#             # Heun's method
#             x_2 = x + d * dt
#             if t_converter is not None:
#                 denoised_2 = model(x_2, cond, sigmas[i + 1] * s_in)
#             else:
#                 denoised_2 = model(x_2, cond, sigmas[i + 1] * s_in)
#             d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
#             d_prime = (d + d_2) / 2
#             x = x + d_prime * dt
#     return x
@torch.no_grad()
def sample_heun(
    model,
    x,
    cond,
    sigmas,
    t_converter=None,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    model.eval()
    extra_args = {"sigma_to_t_callable": t_converter} if extra_args is None else extra_args

    # ---- Guard rails: make sure x/cond are what the UNet expects ----
    # If cond is (cond_img, coords), unwrap the image-like tensor
    _cond_img = cond[0] if isinstance(cond, (tuple, list)) and torch.is_tensor(cond[0]) and cond[0].dim() == 4 else cond

    assert torch.is_tensor(x) and x.dim() == 4, \
        f"[sample_heun] Expected x NCHW; got {type(x)} {None if not torch.is_tensor(x) else tuple(x.shape)}"
    assert torch.is_tensor(_cond_img) and _cond_img.dim() == 4, \
        f"[sample_heun] Expected cond image NCHW; got {type(cond)}"

    # Resolve model.out_channels robustly (Lightning wrapper -> denoiser -> UNet), fallback to target-like x
    out_ch = getattr(model, "out_channels", None)
    if out_ch is None:
        mdl = getattr(model, "model", None)  # LightningDiffusion.model (e.g., EDMDenoiser)
        out_ch = getattr(mdl, "out_channels", None) if mdl is not None else None
        if out_ch is None and mdl is not None:
            inner = getattr(mdl, "inner_model", None)  # EDMDenoiser.inner_model (UNet)
            out_ch = getattr(inner, "out_channels", None) if inner is not None else None
    if out_ch is None:
        out_ch = x.shape[1]

    assert x.shape[1] == out_ch, f"[sample_heun] x has {x.shape[1]} channels, expected {out_ch}"
    assert x.shape[2:] == _cond_img.shape[2:], \
        f"[sample_heun] spatial mismatch: x{tuple(x.shape)} vs cond{tuple(_cond_img.shape)}"

    # ---- Sampling loop ----
    # Ensure s_in matches x device/dtype
    s_in = torch.ones([x.shape[0]], device=x.device, dtype=x.dtype)

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)

        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2).clamp_min(0) ** 0.5

        # Forward passes
        denoised = model(x, cond, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler step
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, cond, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = 0.5 * (d + d_2)
            x = x + d_prime * dt

    return x


@torch.no_grad()
def sample_dpm_2(
    model,
    x,
    cond,
    sigmas,
    sde_type="edm",
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    to_d, s, sigma_inv = get_to_d(sde_type)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = (
                s(sigma_hat) / s(sigmas[i]) * x
                + s(sigma_hat)
                * eps
                * (sigma_hat**2 - sigmas[i] ** 2).clip(min=0) ** 0.5
            )
        denoised = model(x / s(sigma_hat), cond, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigma_inv(sigmas[i + 1]) - sigma_inv(sigma_hat)
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_inv(sigma_mid) - sigma_inv(sigma_hat)
            dt_2 = sigma_inv(sigmas[i + 1]) - sigma_inv(sigma_hat)
            x_2 = x + d * dt_1
            denoised_2 = model(x_2 / s(sigma_mid), cond, sigma_mid * s_in)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2

    return x


@torch.no_grad()
def sample_dpm_2_conditional(
    model,
    x,
    cond,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # x = x.requires_grad_()
        # x_prev = x
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, cond, sigma_hat * s_in, dt=dt)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            denoised = model(x, cond, sigma_hat * s_in, dt=dt)
            d = to_d(x, sigma_hat, denoised)
            x = x + d * dt
        else:
            # DPM-Solver-2
            dt_1 = sigma_mid - sigma_hat
            denoised = model(x, cond, sigma_hat * s_in, dt=dt_1)
            d = to_d(x, sigma_hat, denoised)
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, cond, sigma_mid * s_in, dt=dt_2)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        # if 'posterior_sampler' in extra_args.keys() and sigmas[i + 1] != 0:
        #     posterior_sampler = extra_args['posterior_sampler']
        #     weighted_average_score = (d * dt_1 + d_2 * dt_2)/torch.norm(dt_1 + dt_2)
        #     x = posterior_sampler.apply_conditioning(weighted_average_score, x, sigma_hat, extra_args['measurement'], x_prev)
        #     x = x.detach_()
    return x
