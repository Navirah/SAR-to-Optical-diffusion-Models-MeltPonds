import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from lightning.pytorch import Callback

from .viz_transforms import s1_gray_vis, s2_to_rgb01_from_norm
from .viz_helpers import extract_s1_s2, cat_horiz


class DiffusionDebugCallback(Callback):
    def __init__(
        self,
        scheduler,
        every_n_epochs=1,
        save_dir=None,
        n_samples=8,
        steps=30,
        seed=1234,
        cfg_w=None,
        predicts="x0",
        use_ema=True,
    ):
        self.scheduler = scheduler
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.steps = steps
        self.seed = seed
        self.cfg_w = cfg_w
        self.predicts = predicts
        self.use_ema = use_ema
        """
        Lightning callback for qualitative monitoring of a conditional diffusion model.

        At the end of selected validation epochs, this callback:
        - runs the diffusion sampler on a small validation batch
        - conditions on Sentinel-1 inputs (S1)
        - generates Sentinel-2 predictions (S2)
        - saves a side-by-side visual comparison:
            [ S1 input | S2 ground truth | S2 prediction ]
        """
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch or 0
        if epoch % self.every_n_epochs != 0:
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return

        model = pl_module
        if not isinstance(model, nn.Module):
            return

        was_training = model.training
        ema_ctx = None

        try:
            if self.use_ema and hasattr(model, "use_ema"):
                ema_ctx = model.use_ema()
                ema_ctx.__enter__()

            model.eval()
            torch.manual_seed(self.seed)

            val_loader = trainer.val_dataloaders[0]
            batch = next(iter(val_loader))

            s1, s2 = extract_s1_s2(batch, model.device)
            B = min(self.n_samples, s2.size(0))

            x_pred = self.scheduler.sample(
                model,
                shape=s2[:B].shape,
                cond=s1[:B],
                steps=self.steps,
                cfg_w=self.cfg_w,
                clamp_x0=False,
                seed=self.seed,
            )

            s1_vis = s1_gray_vis(s1[:B]).repeat(1, 3, 1, 1)
            gt_vis = s2_to_rgb01_from_norm(s2[:B])
            pr_vis = s2_to_rgb01_from_norm(x_pred)

            panel = cat_horiz([s1_vis, gt_vis, pr_vis])

            os.makedirs(self.save_dir, exist_ok=True)
            fname = os.path.join(self.save_dir, f"preview_e{epoch:04d}.png")
            save_image(panel, fname, nrow=1)

        finally:
            if was_training:
                model.train()
            if ema_ctx is not None:
                ema_ctx.__exit__(None, None, None)
