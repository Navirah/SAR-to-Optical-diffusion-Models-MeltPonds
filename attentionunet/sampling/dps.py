"""Diffusion posterior sampling methods. 
Here we adapt diffusion posterior sampling methods (https://arxiv.org/pdf/2209.14687).

Code adapted directly from https://github.com/DPS2022/diffusion-posterior-sampling
Main modifications required for the DDIM/Karras EDM formulation.

Code currently NOT BEING USED as we aren't solving inverse problems yet. 
"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from .k_sampling import append_dims, to_d


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


class InpaintingOperator(LinearOperator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, data, **kwargs):
        return data * self.mask

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, **kwargs):
        self.operator = operator

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        # gaussian noise rule
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, **kwargs):
        super().__init__(operator)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        score_modifier = norm_grad * self.scale
        return score_modifier, norm


class ConditionalPosteriorSampler(nn.Module):

    def __init__(
        self, model, conditioner, measurement, sigma_t_callable, noise_t_callable=None
    ):
        super().__init__()

        self.score_model = model
        self.conditioner = conditioner
        self.measurement = measurement
        self.sigma_t_callable = sigma_t_callable
        self.noise_t_callable = lambda t: (
            torch.ones_like(t, device=t.device, dtype=t.dtype)
            if noise_t_callable is None
            else noise_t_callable
        )

    def compute_x0_hat(self, model_output, xt, t):
        sigma_sqr = self.sigma_t_callable(t) ** 2
        noise = self.noise_t_callable(t)
        return sigma_sqr * noise * model_output + noise * xt

    def apply_conditioning(self, unconditional_score, xt, t, x_prev):
        x0_hat = self.compute_x0_hat(unconditional_score, xt, t)

        conditional_score_correction, _ = self.conditioner.conditioning(
            x_prev, x0_hat, self.measurement
        )
        return unconditional_score - conditional_score_correction

    def forward(self, x, cond, sigma, dt, **kwargs):
        with torch.enable_grad():
            x = x.requires_grad_()
            unconditional_score = self.score_model(x, cond, sigma, **kwargs)
            sigma = append_dims(sigma, x.ndim)
            d = to_d(x, sigma, unconditional_score)
            xt = x + d * dt
            # x0_hat = self.compute_x0_hat(score, x, sigma)
            conditional_score = self.apply_conditioning(
                unconditional_score, xt, sigma, x
            )
            # score.backward()
            # print(x.grad)
            # x0_hat = self.compute_x0_hat(score, x, t)
            # difference = self.measurement - self.conditioner.operator.forward(x0_hat)
            # norm = torch.linalg.norm(difference)
            # likelihood_score = -torch.autograd.grad(norm, x)[0] * self.conditioner.scale
            # hat_x0 = x0_hat + sigma.pow(2) * likelihood_score
        return conditional_score
