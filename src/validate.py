# -*- coding: utf-8 -*-
"""
Model validation/inference over folders of NPZ files.

This reproduces the notebook behavior:
- For each "*test" folder under root_dir:
  - For each NPZ file:
    - reshape arr_0 -> (7,7,time)
    - sliding windows of length=10 along time
    - per-window z-score
    - model.predict -> per-window class
    - aggregate per-file by majority vote
  - report folder counts for class0/class1
"""
from __future__ import annotations

import os
import random
import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from tensorflow.keras.models import load_model


@dataclass
class ValidateConfig:
    root_dir: str
    folder_suffix: str = "test"
    file_fraction: float = 1.0
    max_frame: int = 6000
    frame: int = 10
    seed: int = 0


def _sliding_windows_7x7xT(arr_7x7xT: np.ndarray, frame: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(arr_7x7xT, window_shape=frame, axis=2)
    # windows shape: (7,7,num_windows,frame) -> (num_windows,7,7,frame)
    return np.transpose(windows, axes=[2, 0, 1, 3])


def _zscore_per_window(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # windows: (N,7,7,frame)
    mean = np.mean(windows, axis=(1, 2, 3), keepdims=True)
    std = np.std(windows, axis=(1, 2, 3), keepdims=True) + eps
    return (windows - mean) / std


def validate_folders(model, cfg: ValidateConfig) -> List[Dict[str, int]]:
    rng = random.Random(cfg.seed)

    crop_folders = [
        f.path for f in os.scandir(cfg.root_dir)
        if f.is_dir() and f.name.endswith(cfg.folder_suffix)
    ]

    summary: List[Dict[str, int]] = []
    for crop_dir in crop_folders:
        folder_name = os.path.basename(crop_dir)
        file_list = [f for f in os.listdir(crop_dir) if f.lower().endswith(".npz")]
        if not file_list:
            continue

        n_select = max(1, int(len(file_list) * float(cfg.file_fraction)))
        selected_files = rng.sample(file_list, n_select)

        results = np.full((cfg.max_frame, len(selected_files)), 4, dtype=np.int64)  # 4=invalid

        for idx, filename in enumerate(selected_files):
            file_path = os.path.join(crop_dir, filename)
            try:
                raw_data = np.load(file_path)["arr_0"]
            except Exception:
                continue

            try:
                reshaped = np.reshape(raw_data, (7, 7, -1))
            except Exception:
                continue

            try:
                windowed = _sliding_windows_7x7xT(reshaped, cfg.frame)  # (num,7,7,frame)
            except Exception:
                continue
            if windowed.shape[0] == 0:
                continue

            normalized = _zscore_per_window(windowed)[..., np.newaxis]  # (num,7,7,frame,1)

            try:
                preds = model.predict(normalized, verbose=0)
                pred_labels = np.argmax(preds, axis=1).astype(np.int64)
                results[:len(pred_labels), idx] = pred_labels
            except Exception:
                continue

        count0 = 0
        count1 = 0
        for i in range(results.shape[1]):
            preds = results[:, i]
            valid = preds[preds != 4]
            if len(valid) == 0:
                continue
            most_common, _ = collections.Counter(valid).most_common(1)[0]
            if most_common == 0:
                count0 += 1
            elif most_common == 1:
                count1 += 1

        summary.append({"folder": folder_name, "total": len(selected_files), "class_0": count0, "class_1": count1})

    return summary


def load_model_and_validate(model_path: str, cfg: ValidateConfig) -> List[Dict[str, int]]:
    model = load_model(model_path)
    return validate_folders(model, cfg)
