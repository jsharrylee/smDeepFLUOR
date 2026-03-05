# -*- coding: utf-8 -*-
"""
Model definition for smDeepFLUOR (3D CNN).

Matches the notebook architecture:
Input (7,7,10,1) ->
Conv3D(32, (2,2,1), relu) + BN + Dropout(0.1) ->
Conv3D(16, (3,2,1), relu) + BN ->
Conv3D(32, (2,2,10), relu) + BN + Dropout(0.1) ->
Flatten ->
Dense(16, relu, L2=0.05) ->
Dense(2, softmax)
"""
from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


@dataclass
class ModelConfig:
    input_shape: tuple = (7, 7, 10, 1)
    l2: float = 0.05
    dropout1: float = 0.1
    dropout2: float = 0.1


def build_model(cfg: ModelConfig = ModelConfig()) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=cfg.input_shape),
        layers.Conv3D(32, (2, 2, 1), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(cfg.dropout1),

        layers.Conv3D(16, (3, 2, 1), activation="relu"),
        layers.BatchNormalization(),

        layers.Conv3D(32, (2, 2, 10), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(cfg.dropout2),

        layers.Flatten(),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(cfg.l2)),
        layers.Dense(2, activation="softmax"),
    ])
    return model


@dataclass
class CompileConfig:
    learning_rate: float = 1e-5


def compile_model(model: tf.keras.Model, cfg: CompileConfig = CompileConfig()) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg.learning_rate)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model
