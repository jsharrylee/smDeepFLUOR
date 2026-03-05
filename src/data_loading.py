# -*- coding: utf-8 -*-
"""
NPZ loader + sliding-window generator for smDeepFLUOR.

This consolidates the duplicated "Data loading (classA/classB)" notebook cells.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np


@dataclass
class LoadConfig:
    folder_path: str
    use_fraction: float = 0.7      # fraction of sliding windows per file
    file_fraction: float = 1.0     # fraction of files sampled from the folder
    seed: int = 0
    frame: int = 10                # window length


def load_npz_from_folder(cfg: LoadConfig) -> Optional[np.ndarray]:
    """
    Loads .npz files from a folder, applies sliding window (length=cfg.frame),
    subsamples windows per file, and concatenates across files.

    Expects each NPZ has key 'arr_0' with shape: (1, 7, 7, N)
    Returns array with shape: (num_samples, 7, 7, frame)
    """
    rng = random.Random(cfg.seed)

    all_files = [f for f in os.listdir(cfg.folder_path) if f.lower().endswith(".npz")]
    if not all_files:
        return None

    num_files_to_use = max(1, int(len(all_files) * float(cfg.file_fraction)))
    selected_files = rng.sample(all_files, num_files_to_use)

    all_chunks = []
    for filename in selected_files:
        file_path = os.path.join(cfg.folder_path, filename)

        try:
            arr = np.load(file_path)["arr_0"]  # (1,7,7,N)
        except Exception:
            continue

        if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1:3] != (7, 7):
            continue

        arr = arr[0]  # (7,7,N)
        N = arr.shape[2]
        if N < cfg.frame:
            continue

        num_windows = N - cfg.frame + 1
        windows = np.stack([arr[:, :, i:i + cfg.frame] for i in range(num_windows)], axis=0)

        select_n = max(1, int(num_windows * float(cfg.use_fraction)))
        selected_indices = sorted(rng.sample(range(num_windows), select_n))
        selected_windows = windows[selected_indices]

        all_chunks.append(selected_windows)

    if not all_chunks:
        return None

    return np.concatenate(all_chunks, axis=0)


def load_two_classes(
    classA_folder: str,
    classB_folder: str,
    use_fraction: float = 0.7,
    file_fraction: float = 1.0,
    seed: int = 0,
    frame: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to load class A and class B arrays with identical settings.
    """
    A = load_npz_from_folder(LoadConfig(classA_folder, use_fraction, file_fraction, seed, frame))
    B = load_npz_from_folder(LoadConfig(classB_folder, use_fraction, file_fraction, seed + 1, frame))
    if A is None or B is None:
        raise RuntimeError("Failed to load data: one of the class folders returned no usable samples.")
    return A, B
