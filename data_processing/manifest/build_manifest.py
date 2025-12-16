from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

@dataclass(frozen=True)
class ManifestConfig:
    """
    Configuration for building the collocated Sentinel-1 / Sentinel-2 manifest.
    """
    processed_dir: str
    meltpond_list_path: str
    output_csv: str = "selected_processed.csv"

SENTINEL2_REGEX = re.compile(
    r"(S2A_MSIL1C_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{8}T\d{6})"
)

def extract_sentinel2_id(text: str) -> str | None:
    """
    Extract Sentinel-2 product identifier from a string.
    """
    match = SENTINEL2_REGEX.search(text)
    return match.group(1) if match else None

def load_meltpond_file_list(path: str) -> List[str]:
    """
    Load list of MPF NetCDF file paths.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def build_sentinel2_lookup(files: List[str]) -> Dict[str, str]:
    """
    Build lookup from Sentinel-2 identifier to MPF NetCDF file path.
    """
    lookup = {}

    for fp in files:
        sid = extract_sentinel2_id(fp)
        if sid:
            lookup[sid] = fp

    return lookup

def build_manifest_rows(
    processed_dir: str,
    s2_lookup: Dict[str, str],
) -> List[dict]:
    """
    Match processed collocation folders with MPF NetCDF products.

    Multiple processed collocations (e.g. multiple Sentinel-1 overpasses)
    are aggregated per Sentinel-2 acquisition.
    """
    grouped: Dict[str, List[str]] = {}

    for folder_name in sorted(os.listdir(processed_dir)):
        folder_path = os.path.join(processed_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        sentinel2_id = extract_sentinel2_id(folder_name)
        if sentinel2_id is None:
            continue

        if sentinel2_id not in s2_lookup:
            continue

        grouped.setdefault(sentinel2_id, []).append(folder_path)

    rows = []
    for sentinel2_id, folders in grouped.items():
        rows.append({
            "file": s2_lookup[sentinel2_id],
            "collocated_folders": folders,   # ← list of paths
            "sentinel2": sentinel2_id,
            "sentinel1": [os.path.basename(f).split("__")[0] for f in folders],
        })

    return rows


def write_missing_mpf_files(
    all_files: List[str],
    matched_files: List[str],
    output_path: str = "missing_from_processed.txt",
):
    """
    Write MPF files that could not be matched to processed collocations.
    """
    missing = [f for f in all_files if f not in matched_files]

    if not missing:
        return

    with open(output_path, "w") as f:
        for m in missing:
            f.write(m + "\n")

def build_manifest(cfg: ManifestConfig) -> pd.DataFrame:
    """
    Build and save the collocation manifest CSV.

    The resulting CSV links:
    - MPF NetCDF files (with detected melt ponds)
    - processed Sentinel-1 / Sentinel-2 collocation folders
    """
    mpf_files = load_meltpond_file_list(cfg.meltpond_list_path)
    s2_lookup = build_sentinel2_lookup(mpf_files)

    rows = build_manifest_rows(cfg.processed_dir, s2_lookup)
    df = pd.DataFrame(rows)

    df.to_csv(cfg.output_csv, index=False)

    write_missing_mpf_files(
        all_files=mpf_files,
        matched_files=df["file"].tolist(),
    )

    print(f"Processed folders scanned: {len(os.listdir(cfg.processed_dir))}")
    print(f"Manifest rows written: {len(df)}")
    print(f"Output CSV: {cfg.output_csv}")

    return df

if __name__ == "__main__":
    cfg = ManifestConfig(
        processed_dir="/cpnet/li3_cpdata/SATS/COLOCATION_S1_S2/aa533193a035/processing_output",
        meltpond_list_path="artifacts/files_with_melt_ponds.txt",
        output_csv="artifacts/selected_processed.csv",
    )

    build_manifest(cfg)
