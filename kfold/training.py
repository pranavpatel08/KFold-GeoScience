"""Training utilities for KFold stacking experiments."""

from typing import Dict, Tuple, Optional

import utils  # noqa: F401 — must precede tensorflow import to silence TF logging
from utils import log

import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
import models


def make_callbacks():
    return [
        EarlyStopping(**config.EARLY_STOPPING),
        ReduceLROnPlateau(**config.REDUCE_LR),
    ]


def _reshape_for_cnn_rnn(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, X.shape[1], 1)


def generate_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = config.N_FOLDS,
    seed: int = config.GLOBAL_SEED,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Generate out-of-fold predictions using KFold or GroupKFold.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_folds: Number of folds for cross-validation
        seed: Random seed for reproducibility
        groups: Well IDs for well-based group CV (if None, uses standard KFold)

    Returns:
        Dictionary of OOF predictions for each model
    """
    # Use GroupKFold for well-based CV, otherwise standard KFold
    if groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        splits = kf.split(X, y, groups=groups)
        log(f"Performing {n_folds}-fold WELL-BASED cross-validation...")
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(X)
        log(f"Performing {n_folds}-fold cross-validation...")

    oof_preds = {name: np.zeros(len(X)) for name in config.MODEL_ORDER}
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Log well composition if using well-based CV
        if groups is not None:
            val_wells = np.unique(groups[val_idx])
            train_wells = np.unique(groups[train_idx])
            log(f"Fold {fold + 1}/{n_folds} - {len(train_wells)} train wells, {len(val_wells)} val wells")
            log(f"  Val wells: {list(val_wells)}")
        else:
            log(f"Fold {fold + 1}/{n_folds} - preparing data...")
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        dfnn_scaler = RobustScaler()
        X_train_dfnn = dfnn_scaler.fit_transform(X_train)
        X_val_dfnn = dfnn_scaler.transform(X_val)

        X_train_cnn = _reshape_for_cnn_rnn(X_train_s)
        X_val_cnn = _reshape_for_cnn_rnn(X_val_s)

        y_clip_low, y_clip_high = np.percentile(
            y_train,
            [config.Y_CLIP_LOWER_PCT, config.Y_CLIP_UPPER_PCT],
        )
        y_train_clipped = np.clip(y_train, y_clip_low, y_clip_high)

        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train_clipped.reshape(-1, 1)).ravel()

        builders = models.get_model_builders(input_dim=X.shape[1])

        log(f"Fold {fold + 1}/{n_folds} - training CNN...")
        tf.keras.backend.clear_session()
        cnn = builders["CNN"]()
        callbacks = make_callbacks()
        cnn.fit(
            X_train_cnn,
            y_train_s,
            verbose=0,
            callbacks=callbacks,
            **config.CNN_TRAINING,
        )
        cnn_preds = cnn.predict(X_val_cnn, verbose=0).flatten()
        cnn_preds = y_scaler.inverse_transform(cnn_preds.reshape(-1, 1)).ravel()
        oof_preds["CNN"][val_idx] = cnn_preds

        log(f"Fold {fold + 1}/{n_folds} - training DFNN...")
        tf.keras.backend.clear_session()
        dfnn = builders["DFNN"]()
        callbacks = make_callbacks()
        dfnn.fit(
            X_train_dfnn,
            y_train_s,
            verbose=0,
            callbacks=callbacks,
            **config.DFNN_TRAINING,
        )
        dfnn_preds = dfnn.predict(X_val_dfnn, verbose=0).flatten()
        dfnn_preds = y_scaler.inverse_transform(dfnn_preds.reshape(-1, 1)).ravel()
        oof_preds["DFNN"][val_idx] = dfnn_preds

        log(f"Fold {fold + 1}/{n_folds} - training RNN...")
        tf.keras.backend.clear_session()
        rnn = builders["RNN"]()
        callbacks = make_callbacks()
        rnn.fit(
            X_train_cnn,
            y_train_s,
            verbose=0,
            callbacks=callbacks,
            **config.RNN_TRAINING,
        )
        rnn_preds = rnn.predict(X_val_cnn, verbose=0).flatten()
        rnn_preds = y_scaler.inverse_transform(rnn_preds.reshape(-1, 1)).ravel()
        oof_preds["RNN"][val_idx] = rnn_preds

        log(f"Fold {fold + 1}/{n_folds} - training RF...")
        rf = builders["RF"]()
        rf.fit(X_train_s, y_train)
        oof_preds["RF"][val_idx] = rf.predict(X_val_s)

        log(f"Fold {fold + 1}/{n_folds} - training XGBoost...")
        xgb_model = builders["XGBoost"]()
        xgb_model.set_params(early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS)
        xgb_model.fit(
            X_train_s,
            y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )
        oof_preds["XGBoost"][val_idx] = xgb_model.predict(X_val_s)

        fold_metrics = {
            "CNN": r2_score(y_val, oof_preds["CNN"][val_idx]),
            "DFNN": r2_score(y_val, oof_preds["DFNN"][val_idx]),
            "RNN": r2_score(y_val, oof_preds["RNN"][val_idx]),
            "RF": r2_score(y_val, oof_preds["RF"][val_idx]),
            "XGBoost": r2_score(y_val, oof_preds["XGBoost"][val_idx]),
        }
        log(
            "Fold "
            f"{fold + 1}/{n_folds} R2: "
            f"CNN={fold_metrics['CNN']:.4f}, "
            f"DFNN={fold_metrics['DFNN']:.4f}, "
            f"RNN={fold_metrics['RNN']:.4f}, "
            f"RF={fold_metrics['RF']:.4f}, "
            f"XGBoost={fold_metrics['XGBoost']:.4f}"
        )

    return oof_preds


def train_full_models(
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    feature_set: str = "all_features",
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], StandardScaler, Dict[str, np.ndarray]]:
    cache_dir = config.MODEL_CACHE_DIR / feature_set
    cache_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = cache_dir / "x_scaler.joblib"
    y_scaler_path = cache_dir / "y_scaler.joblib"
    dfnn_scaler_path = cache_dir / "dfnn_scaler.joblib"
    cnn_path = cache_dir / "cnn.keras"
    dfnn_path = cache_dir / "dfnn.keras"
    rnn_path = cache_dir / "rnn.keras"
    rf_path = cache_dir / "rf.joblib"
    xgb_path = cache_dir / "xgb.joblib"

    all_cached = all(
        path.exists()
        for path in [
            scaler_path,
            y_scaler_path,
            dfnn_scaler_path,
            cnn_path,
            dfnn_path,
            rnn_path,
            rf_path,
            xgb_path,
        ]
    )

    if all_cached:
        log("Loading cached models and scalers...")
        scaler = joblib.load(scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        X_train_val_s = scaler.transform(X_train_val)
        X_test_s = scaler.transform(X_test)
        dfnn_scaler = joblib.load(dfnn_scaler_path)
        X_train_val_dfnn = dfnn_scaler.transform(X_train_val)
        X_test_dfnn = dfnn_scaler.transform(X_test)

        X_train_val_cnn = _reshape_for_cnn_rnn(X_train_val_s)
        X_test_cnn = _reshape_for_cnn_rnn(X_test_s)

        trained_models: Dict[str, object] = {
            "CNN": tf.keras.models.load_model(cnn_path),
            "DFNN": tf.keras.models.load_model(dfnn_path),
            "RNN": tf.keras.models.load_model(rnn_path),
            "RF": joblib.load(rf_path),
            "XGBoost": joblib.load(xgb_path),
        }

        test_predictions: Dict[str, np.ndarray] = {}
        cnn_preds = trained_models["CNN"].predict(X_test_cnn, verbose=0).flatten()
        test_predictions["CNN"] = y_scaler.inverse_transform(cnn_preds.reshape(-1, 1)).ravel()
        dfnn_preds = trained_models["DFNN"].predict(X_test_dfnn, verbose=0).flatten()
        test_predictions["DFNN"] = y_scaler.inverse_transform(dfnn_preds.reshape(-1, 1)).ravel()
        rnn_preds = trained_models["RNN"].predict(X_test_cnn, verbose=0).flatten()
        test_predictions["RNN"] = y_scaler.inverse_transform(rnn_preds.reshape(-1, 1)).ravel()
        test_predictions["RF"] = trained_models["RF"].predict(X_test_s)
        test_predictions["XGBoost"] = trained_models["XGBoost"].predict(X_test_s)

        inputs = {
            "tabular": X_test_s,
            "sequence": X_test_cnn,
        }
        return trained_models, test_predictions, scaler, inputs

    scaler = StandardScaler()
    X_train_val_s = scaler.fit_transform(X_train_val)
    X_test_s = scaler.transform(X_test)
    dfnn_scaler = RobustScaler()
    X_train_val_dfnn = dfnn_scaler.fit_transform(X_train_val)
    X_test_dfnn = dfnn_scaler.transform(X_test)

    X_train_val_cnn = _reshape_for_cnn_rnn(X_train_val_s)
    X_test_cnn = _reshape_for_cnn_rnn(X_test_s)

    builders = models.get_model_builders(input_dim=X_train_val.shape[1])
    trained_models: Dict[str, object] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    log("Retraining all models on full train+val data...")

    log("  CNN...")
    tf.keras.backend.clear_session()
    cnn = builders["CNN"]()
    callbacks = make_callbacks()
    y_scaler = StandardScaler()
    y_clip_low, y_clip_high = np.percentile(
        y_train_val,
        [config.Y_CLIP_LOWER_PCT, config.Y_CLIP_UPPER_PCT],
    )
    y_train_clipped = np.clip(y_train_val, y_clip_low, y_clip_high)
    y_train_s = y_scaler.fit_transform(y_train_clipped.reshape(-1, 1)).ravel()
    cnn.fit(
        X_train_val_cnn,
        y_train_s,
        verbose=0,
        callbacks=callbacks,
        **config.CNN_TRAINING,
    )
    trained_models["CNN"] = cnn
    cnn_preds = cnn.predict(X_test_cnn, verbose=0).flatten()
    test_predictions["CNN"] = y_scaler.inverse_transform(cnn_preds.reshape(-1, 1)).ravel()

    log("  DFNN...")
    tf.keras.backend.clear_session()
    dfnn = builders["DFNN"]()
    callbacks = make_callbacks()
    dfnn.fit(
        X_train_val_dfnn,
        y_train_s,
        verbose=0,
        callbacks=callbacks,
        **config.DFNN_TRAINING,
    )
    trained_models["DFNN"] = dfnn
    dfnn_preds = dfnn.predict(X_test_dfnn, verbose=0).flatten()
    test_predictions["DFNN"] = y_scaler.inverse_transform(dfnn_preds.reshape(-1, 1)).ravel()

    log("  RNN...")
    tf.keras.backend.clear_session()
    rnn = builders["RNN"]()
    callbacks = make_callbacks()
    rnn.fit(
        X_train_val_cnn,
        y_train_s,
        verbose=0,
        callbacks=callbacks,
        **config.RNN_TRAINING,
    )
    trained_models["RNN"] = rnn
    rnn_preds = rnn.predict(X_test_cnn, verbose=0).flatten()
    test_predictions["RNN"] = y_scaler.inverse_transform(rnn_preds.reshape(-1, 1)).ravel()

    log("  RF...")
    rf = builders["RF"]()
    rf.fit(X_train_val_s, y_train_val)
    trained_models["RF"] = rf
    test_predictions["RF"] = rf.predict(X_test_s)

    log("  XGBoost...")
    xgb_model = builders["XGBoost"]()
    xgb_model.fit(X_train_val_s, y_train_val, verbose=False)
    trained_models["XGBoost"] = xgb_model
    test_predictions["XGBoost"] = xgb_model.predict(X_test_s)

    inputs = {
        "tabular": X_test_s,
        "sequence": X_test_cnn,
    }
    joblib.dump(scaler, scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    joblib.dump(dfnn_scaler, dfnn_scaler_path)
    cnn.save(cnn_path)
    dfnn.save(dfnn_path)
    rnn.save(rnn_path)
    joblib.dump(rf, rf_path)
    joblib.dump(xgb_model, xgb_path)
    return trained_models, test_predictions, scaler, inputs
