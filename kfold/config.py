"""Configuration for KFold stacking experiments."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_CACHE_DIR = RESULTS_DIR / "models"

GLOBAL_SEED = 42
N_FOLDS = 5

TARGET_COL = "ppp"
DEPTH_COL = "tvd"

METADATA_PATH = ROOT_DIR / "preprocessing" / "preprocessing_metadata.json"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
VAL_DATA_PATH = DATA_DIR / "val_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

RAW_FEATURES = [
    "tvd",
    "dt",
    "dt_nct",
    "gr",
    "sphi",
    "hp",
    "ob",
    "rhob_combined",
    "res_deep",
]
ENGINEERED_FEATURES = [
    "eaton_ratio",
    "hp_gradient",
    "ob_gradient",
    "tvd_normalized",
]
ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

MODEL_ORDER = ["CNN", "DFNN", "RNN", "RF", "XGBoost"]

CNN_TRAINING = {"epochs": 100, "batch_size": 32, "validation_split": 0.1}
DFNN_TRAINING = {"epochs": 100, "batch_size": 64, "validation_split": 0.1}
RNN_TRAINING = {"epochs": 130, "batch_size": 16, "validation_split": 0.1}

EARLY_STOPPING = {"monitor": "val_loss", "patience": 5, "restore_best_weights": True}
REDUCE_LR = {"monitor": "val_loss", "patience": 5, "factor": 0.5, "min_lr": 1e-6}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": GLOBAL_SEED,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": GLOBAL_SEED,
}
XGB_EARLY_STOPPING_ROUNDS = 10

STACKING_ALPHA = 1.0

# DFNN stability settings
DFNN_LR = 1e-4
DFNN_CLIPNORM = 1.0
DFNN_L2 = 1e-4

# Target clipping for NN stability (percentiles)
Y_CLIP_LOWER_PCT = 1.0
Y_CLIP_UPPER_PCT = 99.0
