from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDatasetS1S2(Dataset):
    """
    Dataset for paired Sentinel-1 / Sentinel-2 patches.

    Expected input shapes:
      - s1: (N, H, W, 2)  → HH, HV
      - s2: (N, H, W, 4)  → B2, B3, B4, B8

    Returned tensors:
      - s1: (2, H, W)
      - s2: (4, H, W)
    """

    def __init__(self, s1: np.ndarray, s2: np.ndarray):
        assert s1.ndim == 4, f"s1 must be (N,H,W,C), got {s1.shape}"
        assert s2.ndim == 4, f"s2 must be (N,H,W,C), got {s2.shape}"
        assert s1.shape[0] == s2.shape[0], "s1 and s2 must have same N"

        self.s1 = s1.astype(np.float32)
        self.s2 = s2.astype(np.float32)

    def __len__(self) -> int:
        return self.s1.shape[0]

    def __getitem__(self, idx: int):
        s1 = torch.from_numpy(self.s1[idx]).permute(2, 0, 1)  # (C,H,W)
        s2 = torch.from_numpy(self.s2[idx]).permute(2, 0, 1)

        return {
            "s1": s1,
            "s2": s2,
        }
