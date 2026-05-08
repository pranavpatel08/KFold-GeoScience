"""Data loading utilities for KFold stacking experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import config


def load_metadata(path: Path = config.METADATA_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def resolve_features(feature_set: str) -> List[str]:
    if feature_set == "raw_only":
        return config.RAW_FEATURES
    if feature_set == "all_features":
        return config.ALL_FEATURES
    raise ValueError(f"Unknown feature_set: {feature_set}")


def load_datasets(
    feature_set: str = "all_features",
    train_path: Path = config.TRAIN_DATA_PATH,
    val_path: Path = config.VAL_DATA_PATH,
    test_path: Path = config.TEST_DATA_PATH,
    target_col: str = config.TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load datasets with well_id groups for well-based cross-validation."""
    features = resolve_features(feature_set)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    train_val = pd.concat([train_df, val_df], ignore_index=True)

    test_df = pd.read_csv(test_path)

    X_train_val = train_val[features].values.astype(np.float32)
    y_train_val = train_val[target_col].values.astype(np.float32)
    # Extract well_id for well-based group cross-validation
    well_ids = train_val["well_id"].values

    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    return X_train_val, y_train_val, X_test, y_test, features, well_ids
