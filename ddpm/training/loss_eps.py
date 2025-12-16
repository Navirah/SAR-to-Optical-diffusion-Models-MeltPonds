from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_batch_eps(
    model,
    batch: dict,
    optimizer,
    device,
    scheduler,
    is_train: bool = True,
    epoch: int | None = None,
):
    """
    Single training / validation step for ε-prediction DDPM.
    """

    s1 = batch["s1"].to(device)
    s2 = batch["s2"].to(device)

    B = s2.size(0)

    t = scheduler.sample_timesteps(B) #sample timesteps

    #noise & forward diffusion
    noise = torch.randn_like(s2)
    x_t = scheduler.add_noise(s2, noise, t)

    eps_pred = model(x_t, t, s1) #predict epsilon
    loss = F.mse_loss(eps_pred, noise)  #MSE loss

    if is_train:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out = {
        "loss": float(loss),
        "loss_components": {
            "eps_mse": float(loss),
        },
        "t_mean": float(t.float().mean()),
    }

    return loss, out
