"""Credits: Alex Soulis, UCL"""

from .base import LightningBase
from .karras_diffusion import EDMDenoiser, VPDenoiser


def setup_edm_model(config, score_model, device):
    sigma_data = 0.5
    loss_config = {"buffer_width": config.training.loss_buffer_width}
    score_model = EDMDenoiser(score_model, sigma_data, device=device)
    return score_model, loss_config


def setup_vp_model(config, score_model, device):
    loss_config = {"buffer_width": config.training.loss_buffer_width}
    score_model = VPDenoiser(score_model, device=device)
    return score_model, loss_config


class LightningDiffusion(LightningBase):
    """
    Lightning level implementation of the diffusion model.

    We rely on the diffusion model to provide the forward pass and loss function,
    which makes this class very light.
    """

    def __init__(self, model, loss_config, optimizer_config, **kwargs):
        super().__init__(**kwargs)

        self.model = model

        self.optimizer_config = optimizer_config
        self.loss_function = self.set_loss_function(loss_config)

    def forward(self, *x):
        return self.model(*x)

    def set_loss_function(self, loss_config):
        loss_config = dict(loss_config)

        prediction_buffer_width = loss_config.get("buffer_width")
        if prediction_buffer_width is not None:
            self.model.set_buffer_width(prediction_buffer_width)

        def loss(model, target, condition):
            return self.model.loss(target, condition)

        return loss
