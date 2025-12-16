import inspect
import torch

from sampling import k_sampling as ks


def _karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float, device=None, dtype=None):
    """Local Karras schedule (hi->lo) + append zero, mirroring your ks.get_sigmas_karras."""
    i = torch.linspace(0, 1, steps=n, device=device, dtype=torch.float32)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + i * (min_inv - max_inv)) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    if dtype is not None:
        sigmas = sigmas.to(dtype)
    return sigmas


def _resolve_sampler(name: str):
    name = (name or "").lower()
    cand = {
        "heun": getattr(ks, "sample_heun", None),
        "dpm_2": getattr(ks, "sample_dpm_2", None),
        "euler": getattr(ks, "sample_euler", None),
    }
    f = cand.get(name)
    if callable(f):
        return name, f
    for k in ("heun", "dpm_2", "euler"):
        f = cand.get(k)
        if callable(f):
            return k, f
    raise RuntimeError("No valid sampler found: expected one of {heun,dpm_2,euler} in k_sampling.py")


class KarrasSchedulerAdapter:
    """
    Adapter so code can call:
        .sample(pl_module, shape, cond, steps=None, cfg_w=None, clamp_x0=False, seed=None)

    Matches your sampler signatures:
        sampler(model_fn, x, cond, sigmas, **kwargs)
    """

    def __init__(self, n, rho, sigma_min, sigma_max, sampler="heun", sde_type="edm"):
        self.n = int(n)
        self.rho = float(rho)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sampler_name, self._sampler = _resolve_sampler(sampler)
        # Detect which optional kwargs the sampler supports
        sig = inspect.signature(self._sampler)
        self._has_sde_type = "sde_type" in sig.parameters
        self._has_extra_args = "extra_args" in sig.parameters
        self._has_callback = "callback" in sig.parameters
        self._has_disable = "disable" in sig.parameters
        self._has_s_churn = "s_churn" in sig.parameters
        self._has_s_tmin  = "s_tmin"  in sig.parameters
        self._has_s_tmax  = "s_tmax"  in sig.parameters
        self._has_s_noise = "s_noise" in sig.parameters
        self._has_t_converter = "t_converter" in sig.parameters
        self._default_sde_type = sde_type

    @torch.no_grad()
    def sample(self, pl_module, shape, cond, steps=None, cfg_w=None, clamp_x0=False, seed=None):
        """
        pl_module: LightningModule callable as pl_module(x, cond, sigma, dt=...)
        shape    : (B, C, H, W) target shape (S2)
        cond     : (B, 2, H, W) S1 condition
        steps    : number of Karras steps (default self.n)
        """
        device = cond.device
        steps = int(steps or self.n)

        # init noise
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(int(seed))
            x = torch.randn(shape, generator=g, device=device)
        else:
            x = torch.randn(shape, device=device)

        # build Karras sigmas (hi->lo) + zero
        get_sig = getattr(ks, "get_sigmas_karras", None)
        if callable(get_sig):
            sigmas = get_sig(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, device=device)
        else:
            sigmas = _karras_sigmas(steps, self.sigma_min, self.sigma_max, self.rho, device=device)
        sigmas = sigmas.to(dtype=x.dtype)

        # scale init noise to top sigma
        x = x * sigmas[0]

        # model wrapper matches sampler's expectation: f(x, cond, sigma, **kwargs)
        def model_fn(x_in, cond_in, sigma, **kwargs):
            return pl_module(x_in, cond_in, sigma, **kwargs)

        # Common kwargs your samplers accept; pass only those supported
        kwargs = {}
        if self._has_sde_type:
            kwargs["sde_type"] = self._default_sde_type
        if self._has_extra_args:
            kwargs["extra_args"] = {}   # not used in your current functions, but harmless
        if self._has_disable:
            kwargs["disable"] = True
        if self._has_s_churn:
            kwargs["s_churn"] = 0.0
        if self._has_s_tmin:
            kwargs["s_tmin"] = 0.0
        if self._has_s_tmax:
            kwargs["s_tmax"] = float("inf")
        if self._has_s_noise:
            kwargs["s_noise"] = 1.0
        if self._has_t_converter:
            kwargs["t_converter"] = None

        # CRITICAL: call with (model_fn, x, cond, sigmas, **kwargs)
        return self._sampler(model_fn, x, cond, sigmas, **kwargs)
