from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_batch_x0(
    model,
    batch: dict,
    optimizer,
    device,
    scheduler,
    is_train: bool = True,
    epoch: int | None = None,
):
    """
    Single training / validation step for x0-prediction DDPM.

    Expected batch format:
      batch["s1"]: (B, 2, H, W)
      batch["s2"]: (B, 4, H, W)

    Scheduler requirements:
      - sample_timesteps(B)
      - add_noise(x0, noise, t)

    Returns:
      loss (torch.Tensor)
      out  (dict) with logging-friendly components
    """

    s1 = batch["s1"].to(device)
    s2 = batch["s2"].to(device)

    B = s2.size(0)
    
    t = scheduler.sample_timesteps(B) #sample timesteps

    #forward diffusion
    noise = torch.randn_like(s2)
    x_t = scheduler.add_noise(s2, noise, t)

    x0_pred = model(x_t, t, s1) #predict x0 directly
    loss = F.mse_loss(x0_pred, s2) #MSE loss

    if is_train:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out = {
        "loss": float(loss),
        "loss_components": {
            "x0_mse": float(loss),
        },
        "t_mean": float(t.float().mean()),
    }

    return loss, out
