# -*- coding: utf-8 -*-
"""
Data standardization utilities for smDeepFLUOR.

Notebook behavior:
- combine classA and classB arrays: (N,7,7,10)
- label: A=0, B=1
- per-sample z-score over (7,7,10)
- train/test split
- expand dims to (N,7,7,10,1) for Conv3D input
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class StandardizeConfig:
    test_size: float = 0.2
    random_state: int = 42
    eps: float = 1e-8


def standardize_and_split(
    classA: np.ndarray,
    classB: np.ndarray,
    cfg: StandardizeConfig = StandardizeConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Combine and label
    X = np.append(classA, classB, axis=0)
    yA = np.full((classA.shape[0],), 0, dtype=np.int64)
    yB = np.full((classB.shape[0],), 1, dtype=np.int64)
    y = np.append(yA, yB, axis=0)

    # Per-sample z-score
    mean_vals = np.mean(X, axis=(1, 2, 3), keepdims=True)
    std_vals = np.std(X, axis=(1, 2, 3), keepdims=True)
    X = (X - mean_vals) / (std_vals + cfg.eps)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.test_size), random_state=int(cfg.random_state)
    )

    # Expand dims for Conv3D: (N,7,7,10,1)
    if X_train.ndim == 4:
        X_train = X_train[..., np.newaxis]
    if X_test.ndim == 4:
        X_test = X_test[..., np.newaxis]

    return X_train, X_test, y_train, y_test
