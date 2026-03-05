# -*- coding: utf-8 -*-
"""
Single-particle cropping utilities for smDeepFLUOR.

This module implements:
1) Reading Fiji/MOSAIC Particle Tracker CSV
2) Cropping 7x7 ROIs across time
3) Saving per-trajectory crops as TIFF (ImageJ-compatible)
4) Converting cropped TIFFs into NPZ (arr_0: data, arr_1: label)

The logic is intentionally close to the original notebook implementation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import tifffile as tiff
from numpy import savez_compressed

from .io_tiff import read_tiff_stack_TYX


@dataclass
class CroppingConfig:
    csv_file_path: str
    tiff_path: str
    output_dir: str
    csv_encoding: str = "cp949"   # Fiji CSV often in cp949 on KR Windows
    min_traj_len: int = 10
    crop_radius: int = 3          # 7x7 => radius 3
    # Expected original field-of-view size. Used for boundary QC.
    # If you use other dimensions, adjust accordingly.
    fov_min_xy: int = 5
    fov_max_xy: int = 508         # 512-4
    require_t0: bool = True       # keep only trajectories that include t==0


def load_particle_tracker_csv(csv_file_path: str, encoding: str = "cp949") -> pd.DataFrame:
    """
    Load Fiji MOSAIC Particle Tracker table CSV, skipping the first 3 header rows.

    Returns a DataFrame with columns: Tr, x, y, t (float).
    """
    df = pd.read_csv(csv_file_path, encoding=encoding)
    df = df.iloc[3:]  # skip ImageJ header rows

    x = df.iloc[:, 3].to_numpy().astype(float)
    y = df.iloc[:, 4].to_numpy().astype(float)
    tr = df.iloc[:, 1].to_numpy().astype(float)
    tvals = df.iloc[:, 2].to_numpy().astype(float)

    table = pd.DataFrame({"Tr": tr, "x": x, "y": y, "t": tvals})
    table = table.dropna(subset=["Tr"])

    return table


def crop_trajectories_to_tiff(config: CroppingConfig) -> List[str]:
    """
    Crop 7x7 ROIs for each trajectory and save as ImageJ-compatible TIFF.
    Returns list of written TIFF paths.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    table = load_particle_tracker_csv(config.csv_file_path, encoding=config.csv_encoding)
    stack = read_tiff_stack_TYX(config.tiff_path)  # (T,Y,X)

    written: List[str] = []
    r = int(config.crop_radius)

    for tr_id in table["Tr"].unique():
        traj = table[table["Tr"] == tr_id]

        if len(traj) < config.min_traj_len:
            continue

        # Boundary QC (hard-coded for 512x512; configurable via config)
        if (traj["x"] < config.fov_min_xy).any() or (traj["y"] < config.fov_min_xy).any():
            continue
        if (traj["x"] > config.fov_max_xy).any() or (traj["y"] > config.fov_max_xy).any():
            continue

        if config.require_t0 and not (traj["t"] == 0).any():
            continue

        crop_stack: List[np.ndarray] = []

        # Keep original behavior: start from j=1 (skip first row)
        for j in range(1, len(traj)):
            xx = int(traj.x.iloc[j])
            yy = int(traj.y.iloc[j])
            frame_num = int(traj.t.iloc[j])

            if 0 <= frame_num < stack.shape[0]:
                crop = stack[frame_num, yy - r : yy + r + 1, xx - r : xx + r + 1]
                crop_stack.append(crop)

        if not crop_stack:
            continue

        # NOTE: filename uses last (xx,yy) values, consistent with notebook behavior
        output_filename = f"Tr_{int(tr_id)}_x{xx}_y{yy}_t{len(traj)}.tif"
        output_path = os.path.join(config.output_dir, output_filename)

        tiff.imwrite(
            output_path,
            np.asarray(crop_stack, dtype=stack.dtype),
            imagej=True,
            metadata={"axes": "TYX"},
        )
        written.append(output_path)

    return written


def crops_tiff_to_npz(
    crops_dir: str,
    remove_first_frame: bool = True,
    frame_block: int = 10,
    label_value: int = 0,
) -> List[str]:
    """
    Convert cropped TIFFs (T,7,7) into compressed NPZ files.

    Output NPZ:
      - arr_0: shape (1, 7, 7, T)
      - arr_1: labels, shape (1,) with a single integer (default 0)

    Returns list of written NPZ paths.
    """
    file_list = [
        fname for fname in os.listdir(crops_dir)
        if os.path.splitext(fname)[1].lower() in (".tif", ".tiff")
    ]

    written: List[str] = []
    for fname in file_list:
        file_path = os.path.join(crops_dir, fname)
        output = tiff.imread(file_path)  # expected (T,7,7)

        if remove_first_frame and output.shape[0] >= 1:
            output = np.delete(output, 0, axis=0)

        i = len(output) // frame_block
        if i < 1:
            continue

        # Corner-based QC (same as notebook)
        if (any(output[:, 0, 0] != 0) and
            any(output[:, 6, 0] != 0) and
            any(output[:, 0, 6] != 0) and
            any(output[:, 6, 6] != 0)):

            results_sum = np.transpose(output, (1, 2, 0))       # (7,7,T)
            results_sum_resize = results_sum[np.newaxis, ...]   # (1,7,7,T)

            results_4d = np.empty((0, 7, 7, results_sum_resize.shape[-1]), dtype=results_sum_resize.dtype)
            results_4d = np.append(results_4d, results_sum_resize, axis=0)

            y_label_3d = np.full((results_sum_resize.shape[0],), int(label_value))

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(os.path.dirname(file_path), base_name + ".npz")
            savez_compressed(save_path, results_4d, y_label_3d)
            written.append(save_path)

    return written
