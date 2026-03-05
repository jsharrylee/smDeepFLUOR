# -*- coding: utf-8 -*-
"""
Train/test split for smDeepFLUOR NPZ crops.

Replicates the notebook behavior:
- Search for subfolders ending with "-crop" under root_dir
- Collect .npz files in each crop folder
- Random 80/20 split into root_dir/training and root_dir/test
- Prefix destination filenames with "<crop_folder_name>__"
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitConfig:
    root_dir: str
    train_subdir: str = "training"
    test_subdir: str = "test"
    crop_folder_suffix: str = "-crop"
    train_ratio: float = 0.8
    seed: int = 0


def split_npz_train_test(config: SplitConfig) -> Tuple[str, str]:
    random.seed(config.seed)

    train_dir = os.path.join(config.root_dir, config.train_subdir)
    test_dir = os.path.join(config.root_dir, config.test_subdir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    crop_folders = [
        f.path for f in os.scandir(config.root_dir)
        if f.is_dir() and f.name.endswith(config.crop_folder_suffix)
    ]

    for crop_folder_path in crop_folders:
        crop_folder_name = os.path.basename(crop_folder_path)
        npz_files = [f for f in os.listdir(crop_folder_path) if f.lower().endswith(".npz")]
        if not npz_files:
            continue

        random.shuffle(npz_files)
        split_index = int(len(npz_files) * float(config.train_ratio))
        train_files = npz_files[:split_index]
        test_files = npz_files[split_index:]

        def _copy(files: List[str], target_dir: str) -> None:
            for fname in files:
                src = os.path.join(crop_folder_path, fname)
                dst_name = f"{crop_folder_name}__{fname}"
                dst = os.path.join(target_dir, dst_name)
                shutil.copy2(src, dst)

        _copy(train_files, train_dir)
        _copy(test_files, test_dir)

    return train_dir, test_dir
