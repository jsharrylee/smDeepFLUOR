# -*- coding: utf-8 -*-
"""
Training utilities for smDeepFLUOR.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from .model_def import build_model, compile_model, ModelConfig, CompileConfig


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 30
    model_dir: str = "./models"
    monitor: str = "val_accuracy"
    min_lr: float = 1e-6
    reduce_factor: float = 0.5
    reduce_patience: int = 3
    early_patience: int = 6


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    train_cfg: TrainConfig = TrainConfig(),
    model_cfg: ModelConfig = ModelConfig(),
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, str]:
    os.makedirs(train_cfg.model_dir, exist_ok=True)
    model_name = f"best_model_training_bs{train_cfg.batch_size}_lr{train_cfg.learning_rate:.0e}.keras"
    model_path = os.path.join(train_cfg.model_dir, model_name)

    model = build_model(model_cfg)
    model = compile_model(model, CompileConfig(train_cfg.learning_rate))

    checkpoint = ModelCheckpoint(
        model_path,
        monitor=train_cfg.monitor,
        save_best_only=True,
        mode="max" if "acc" in train_cfg.monitor else "min",
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=train_cfg.reduce_factor,
        patience=train_cfg.reduce_patience,
        min_lr=train_cfg.min_lr,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=train_cfg.early_patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=2,
    )

    return model, history, model_path
