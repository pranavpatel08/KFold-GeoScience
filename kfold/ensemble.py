"""Ensembling utilities for KFold stacking experiments."""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge

import config


def compute_meta_features(oof_preds: Dict[str, np.ndarray], model_order: List[str]) -> np.ndarray:
    return np.column_stack([oof_preds[name] for name in model_order])


def train_stacking_model(X_meta: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=config.STACKING_ALPHA)
    model.fit(X_meta, y)
    return model


def rank_models_by_oof(cv_results: Dict[str, Dict[str, float]]) -> List[str]:
    ranked = sorted(cv_results.items(), key=lambda x: x[1]["oof_r2"], reverse=True)
    return [name for name, _ in ranked]


def top_k_ablation(
    model_ranking: List[str],
    test_predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    results = []
    for k in range(1, len(model_ranking) + 1):
        selected = model_ranking[:k]
        selected_preds = np.mean([test_predictions[name] for name in selected], axis=0)
        r2 = float(1 - np.sum((y_test - selected_preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        rmse = float(np.sqrt(np.mean((y_test - selected_preds) ** 2)))
        mae = float(np.mean(np.abs(y_test - selected_preds)))
        results.append(
            {
                "k": k,
                "selected_models": selected,
                "test_r2": r2,
                "test_rmse": rmse,
                "test_mae": mae,
            }
        )
    return results


def drop_one_analysis(
    top_k_models: List[str],
    test_predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    full_preds = np.mean([test_predictions[name] for name in top_k_models], axis=0)
    full_r2 = float(1 - np.sum((y_test - full_preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    results.append(
        {
            "dropped_model": "None",
            "remaining_models": top_k_models,
            "test_r2": full_r2,
            "delta_from_full": "baseline",
        }
    )

    for drop_model in top_k_models:
        remaining = [m for m in top_k_models if m != drop_model]
        preds = np.mean([test_predictions[name] for name in remaining], axis=0)
        r2 = float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        results.append(
            {
                "dropped_model": drop_model,
                "remaining_models": remaining,
                "test_r2": r2,
                "delta_from_full": f"{(r2 - full_r2) * 100:+.2f}%",
            }
        )
    return results


def _oof_ensemble_r2(selected: List[str], oof_preds: Dict[str, np.ndarray], y: np.ndarray) -> float:
    """Compute OOF R² for a simple average ensemble of the selected models."""
    preds = np.mean([oof_preds[name] for name in selected], axis=0)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def greedy_forward_selection(
    model_names: List[str],
    oof_preds: Dict[str, np.ndarray],
    y: np.ndarray,
) -> List[Dict[str, object]]:
    """Greedily add the model that gives the best marginal OOF R² gain.

    Unlike top-K (which adds models in individual-ranking order), this picks
    models by their *marginal contribution* to the current ensemble, naturally
    avoiding redundant models (e.g. it will skip RF if XGBoost is already in).

    Returns a list of steps, each containing:
      - step: int
      - added_model: str  (or None for the stopping row)
      - selected_models: List[str]
      - oof_r2: float
      - delta_r2: float (gain vs previous step)
    """
    remaining = list(model_names)
    selected: List[str] = []
    steps: List[Dict[str, object]] = []
    current_r2 = float("-inf")

    while remaining:
        best_gain = float("-inf")
        best_model = None
        best_r2 = float("-inf")
        for candidate in remaining:
            candidate_r2 = _oof_ensemble_r2(selected + [candidate], oof_preds, y)
            gain = candidate_r2 - (current_r2 if selected else float("-inf"))
            if best_model is None or candidate_r2 > best_r2:
                best_r2 = candidate_r2
                best_gain = candidate_r2 - current_r2 if selected else 0.0
                best_model = candidate

        selected.append(best_model)
        remaining.remove(best_model)
        steps.append(
            {
                "step": len(selected),
                "added_model": best_model,
                "selected_models": list(selected),
                "oof_r2": best_r2,
                "delta_r2": best_r2 - current_r2 if len(selected) > 1 else 0.0,
            }
        )
        current_r2 = best_r2

    return steps


def exhaustive_subset_search(
    model_names: List[str],
    oof_preds: Dict[str, np.ndarray],
    y: np.ndarray,
) -> List[Dict[str, object]]:
    """Evaluate all 2^K - 1 non-empty model subsets on OOF predictions.

    Returns results sorted by oof_r2 descending. Computationally trivial —
    no retraining, just averaging existing OOF predictions.
    """
    results: List[Dict[str, object]] = []
    for size in range(1, len(model_names) + 1):
        for subset in itertools.combinations(model_names, size):
            subset_list = list(subset)
            r2 = _oof_ensemble_r2(subset_list, oof_preds, y)
            results.append(
                {
                    "k": size,
                    "models": subset_list,
                    "oof_r2": r2,
                }
            )
    results.sort(key=lambda x: x["oof_r2"], reverse=True)
    return results


def prediction_correlation_matrix(
    preds: Dict[str, np.ndarray],
    model_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise Pearson correlation between model predictions.

    Returns:
        corr_matrix: (K, K) numpy array
        model_names:  ordered list matching matrix rows/cols
    """
    matrix = np.column_stack([preds[name] for name in model_names])
    corr = np.corrcoef(matrix.T)
    return corr, model_names
