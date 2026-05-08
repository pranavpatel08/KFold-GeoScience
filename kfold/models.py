"""Model builders for KFold stacking experiments."""

from typing import Callable, Dict

import utils  # noqa: F401 — must precede tensorflow import to silence TF logging
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import layers, models, regularizers, optimizers

import config


def build_cnn(input_dim: int):
    model = models.Sequential(
        [
            layers.Input((input_dim, 1)),
            layers.Conv1D(
                64,
                kernel_size=2,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.L2(0.001),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv1D(
                128,
                kernel_size=2,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.L2(0.001),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(0.002)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L2(0.002)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_dfnn(input_dim: int):
    """Deep Feedforward Neural Network.

    Architecture: input_dim → 128 → 64 → 32 → 1
    With 13 features: 13 → 128 → 64 → 32 → 1
    """
    optimizer = optimizers.Adam(learning_rate=config.DFNN_LR, clipnorm=config.DFNN_CLIPNORM)
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(config.DFNN_L2)),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L2(config.DFNN_L2)),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.L2(config.DFNN_L2)),
            layers.Dropout(0.1),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse")
    return model


def build_rnn(input_dim: int):
    model = models.Sequential(
        [
            layers.Input((input_dim, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_random_forest():
    return RandomForestRegressor(**config.RF_PARAMS)


def build_xgboost():
    return xgb.XGBRegressor(**config.XGB_PARAMS)


def get_model_builders(input_dim: int) -> Dict[str, Callable]:
    return {
        "CNN": lambda: build_cnn(input_dim),
        "DFNN": lambda: build_dfnn(input_dim),
        "RNN": lambda: build_rnn(input_dim),
        "RF": build_random_forest,
        "XGBoost": build_xgboost,
    }
