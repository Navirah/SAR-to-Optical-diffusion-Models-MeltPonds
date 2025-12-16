from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import xarray as xr

@dataclass(frozen=True)
class MeltPondDiscoveryConfig:
    """
    Configuration for discovering MPF files containing melt ponds.
    """
    base_dir: str
    output_txt: str = "files_with_melt_ponds.txt"

def has_melt_pond(nc_path: str, var: str = "mpf") -> bool:
    """
    Check whether a NetCDF file contains non-zero melt pond fraction values.
    """
    try:
        with xr.open_dataset(nc_path) as ds:
            if var not in ds:
                return False
            return bool((ds[var] > 0).any())
    except Exception as e:
        print(f"[WARNING] Failed to process {nc_path}: {e}")
        return False

def discover_melt_pond_files(
    base_dir: str,
) -> List[str]:
    """
    Recursively scan a directory for MPF NetCDF files containing melt ponds.
    """
    discovered = []

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".nc"):
                continue

            nc_path = os.path.join(root, fname)
            if has_melt_pond(nc_path):
                discovered.append(nc_path)

    return discovered

def save_file_list(
    files: List[str],
    path: str,
):
    """
    Save list of file paths to a text file.
    """
    with open(path, "w") as f:
        for fp in files:
            f.write(fp + "\n")
def run_discovery(cfg: MeltPondDiscoveryConfig) -> List[str]:
    """
    Run melt pond discovery and persist results.
    """
    files = discover_melt_pond_files(cfg.base_dir)

    print(f"Discovered {len(files)} MPF files with melt ponds.")
    save_file_list(files, cfg.output_txt)

    return files

if __name__ == "__main__":
    cfg = MeltPondDiscoveryConfig(
        base_dir="/home/wch/MPF_Navirah/processed_mpf_collocated",
        output_txt="artifacts/files_with_melt_ponds.txt",
    )

    run_discovery(cfg)
