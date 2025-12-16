"""
Data processing pipeline for Sentinel-1 → Sentinel-2 experiments.

Includes:
- Melt pond discovery
- Collocation manifest building
- Patch extraction
- Dataset preprocessing and normalization
"""

from .preprocess import (
    preprocess_dataframe,
    save_processed_to_npz,
)

__all__ = [
    "preprocess_dataframe",
    "save_processed_to_npz",
]
