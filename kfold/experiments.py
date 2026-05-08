"""Experiment runners for KFold stacking outputs."""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import utils  # noqa: F401 — must precede tensorflow import to silence TF logging
from utils import format_cv_table, get_hardware_info, log, save_csv_dataframe, save_json, set_seeds

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
import data_loader
import ensemble
import training


def log_meta_learner_weights(model: object, model_names: List[str], label: str = "Meta-learner"):
    log(f"\n{label} coefficients:")
    for name, coef in zip(model_names, model.coef_):
        log(f"  {name}: {coef:.4f}")
    log(f"  Intercept: {model.intercept_:.4f}")


def compute_cv_results(oof_preds: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    cv_results: Dict[str, Dict[str, float]] = {}
    for name, preds in oof_preds.items():
        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        cv_results[name] = {"oof_r2": float(r2), "oof_rmse": float(rmse), "oof_mae": float(mae)}
    return cv_results


def _cv_cache_paths(feature_set: str) -> Tuple[str, str]:
    if feature_set == "all_features":
        return (
            str(config.RESULTS_DIR / "cv_results.json"),
            str(config.RESULTS_DIR / "oof_preds_all_features.npz"),
        )
    suffix = feature_set
    return (
        str(config.RESULTS_DIR / f"cv_results_{suffix}.json"),
        str(config.RESULTS_DIR / f"oof_preds_{suffix}.npz"),
    )


def load_cv_results(feature_set: str) -> Dict[str, Dict[str, float]] | None:
    cv_path, _ = _cv_cache_paths(feature_set)
    path = config.RESULTS_DIR / Path(cv_path).name
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_oof_preds(feature_set: str) -> Dict[str, np.ndarray] | None:
    _, oof_path = _cv_cache_paths(feature_set)
    path = config.RESULTS_DIR / Path(oof_path).name
    if not path.exists():
        return None
    data = np.load(path)
    return {name: data[name] for name in data.files}


def run_cv_experiment(
    feature_set: str = "all_features",
    seed: int = config.GLOBAL_SEED,
    save_results: bool = True,
    use_cache: bool = True,
) -> Tuple[Dict, Dict]:
    log("=" * 60)
    log("STEP 1: GENERATING OUT-OF-FOLD PREDICTIONS")
    log("=" * 60)
    set_seeds(seed)

    X_train_val, y_train_val, _, _, _, well_ids = data_loader.load_datasets(feature_set=feature_set)
    if use_cache:
        cached_oof = load_oof_preds(feature_set)
        cached_cv = load_cv_results(feature_set)
        if cached_oof is not None and cached_cv is not None:
            log("Loaded cached OOF predictions and CV results.")
            log("\n" + format_cv_table(cached_cv, config.MODEL_ORDER))
            return cached_oof, cached_cv

    oof_preds = training.generate_oof_predictions(X_train_val, y_train_val, seed=seed, groups=well_ids)

    cv_results = compute_cv_results(oof_preds, y_train_val)
    if save_results:
        cv_path, oof_path = _cv_cache_paths(feature_set)
        save_json(cv_results, Path(cv_path))
        np.savez(oof_path, **oof_preds)

    log("Training stacking meta-learner on OOF predictions...")
    X_meta = ensemble.compute_meta_features(oof_preds, config.MODEL_ORDER)
    stacking_model = ensemble.train_stacking_model(X_meta, y_train_val)
    for name, coef in zip(config.MODEL_ORDER, stacking_model.coef_):
        log(f"  stacking_coef[{name}]: {coef:.4f}")
    log("Note: stacking coefficients are not OOF R2 scores.")

    log("\n" + format_cv_table(cv_results, config.MODEL_ORDER))
    return oof_preds, cv_results


def run_top_k_experiment(
    seed: int = config.GLOBAL_SEED,
    cv_results: Dict[str, Dict[str, float]] | None = None,
    oof_preds: Dict[str, np.ndarray] | None = None,
) -> List[Dict[str, object]]:
    log("=" * 60)
    log("STEP 2: TOP-K ABLATION STUDY")
    log("=" * 60)
    set_seeds(seed)

    X_train_val, y_train_val, X_test, y_test, _, _ = data_loader.load_datasets(feature_set="all_features")
    if oof_preds is None or cv_results is None:
        oof_preds, cv_results = run_cv_experiment(
            feature_set="all_features",
            seed=seed,
            save_results=True,
            use_cache=True,
        )
    model_ranking = ensemble.rank_models_by_oof(cv_results)

    _, test_predictions, _, _ = training.train_full_models(
        X_train_val,
        y_train_val,
        X_test,
        feature_set="all_features",
    )
    top_k_results = ensemble.top_k_ablation(model_ranking, test_predictions, y_test)

    df = pd.DataFrame(top_k_results)
    save_csv_dataframe(df, config.RESULTS_DIR / "top_k_ablation.csv")
    return top_k_results


def run_feature_ablation(
    seed: int = config.GLOBAL_SEED,
) -> pd.DataFrame:
    log("=" * 60)
    log("STEP 3: FEATURE ABLATION STUDY")
    log("=" * 60)
    set_seeds(seed)

    results_rows: List[Dict[str, object]] = []
    _, raw_cv = run_cv_experiment(
        feature_set="raw_only",
        seed=seed,
        save_results=True,
        use_cache=True,
    )
    _, all_cv = run_cv_experiment(
        feature_set="all_features",
        seed=seed,
        save_results=True,
        use_cache=True,
    )

    for model_name in config.MODEL_ORDER:
        raw_stats = raw_cv[model_name]
        all_stats = all_cv[model_name]
        baseline_r2 = raw_stats["oof_r2"]
        delta = all_stats["oof_r2"] - baseline_r2
        delta_pct = f"{delta * 100:+.2f}%"
        results_rows.append(
            {
                "feature_set": "raw_only",
                "model": model_name,
                "oof_r2": raw_stats["oof_r2"],
                "oof_rmse": raw_stats["oof_rmse"],
                "delta_r2": "baseline",
            }
        )
        results_rows.append(
            {
                "feature_set": "all_features",
                "model": model_name,
                "oof_r2": all_stats["oof_r2"],
                "oof_rmse": all_stats["oof_rmse"],
                "delta_r2": delta_pct,
            }
        )

    df = pd.DataFrame(results_rows)
    save_csv_dataframe(df, config.RESULTS_DIR / "feature_ablation.csv")
    return df


def run_drop_one(
    seed: int = config.GLOBAL_SEED,
    cv_results: Dict[str, Dict[str, float]] | None = None,
    oof_preds: Dict[str, np.ndarray] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    log("=" * 60)
    log("STEP 4: DROP-ONE ANALYSIS")
    log("=" * 60)
    set_seeds(seed)

    X_train_val, y_train_val, X_test, y_test, _, _ = data_loader.load_datasets(feature_set="all_features")
    if oof_preds is None or cv_results is None:
        oof_preds, cv_results = run_cv_experiment(
            feature_set="all_features",
            seed=seed,
            save_results=True,
            use_cache=True,
        )
    model_ranking = ensemble.rank_models_by_oof(cv_results)

    _, test_predictions, _, _ = training.train_full_models(
        X_train_val,
        y_train_val,
        X_test,
        feature_set="all_features",
    )

    if top_k is not None:
        best_k = top_k
        log(f"Using provided top-K: {best_k}")
    else:
        top_k_results = ensemble.top_k_ablation(model_ranking, test_predictions, y_test)
        best_k = max(top_k_results, key=lambda x: x["test_r2"])["k"]
        log(f"Automatically determined best-K: {best_k}")

    top_k_models = model_ranking[:best_k]

    drop_one_results = ensemble.drop_one_analysis(top_k_models, test_predictions, y_test)
    df = pd.DataFrame(drop_one_results)
    save_csv_dataframe(df, config.RESULTS_DIR / "drop_one_analysis.csv")
    return df


def benchmark_model(model, X_sample, model_type: str = "keras", n_warmup: int = 10, n_runs: int = 100) -> Dict[str, float]:
    for _ in range(n_warmup):
        if model_type == "keras":
            _ = model.predict(X_sample, verbose=0)
        else:
            _ = model.predict(X_sample)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if model_type == "keras":
            _ = model.predict(X_sample, verbose=0)
        else:
            _ = model.predict(X_sample)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "median_ms": float(np.median(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
        "n_samples": int(len(X_sample)),
    }


def run_benchmark(
    seed: int = config.GLOBAL_SEED,
    batch_size: int = 1000,
) -> Dict[str, object]:
    log("=" * 60)
    log("STEP 5: INFERENCE BENCHMARKING")
    log("=" * 60)
    set_seeds(seed)

    X_train_val, y_train_val, X_test, _, _, _ = data_loader.load_datasets(feature_set="all_features")
    trained_models, _, _, inputs = training.train_full_models(
        X_train_val,
        y_train_val,
        X_test,
        feature_set="all_features",
    )

    single_sample_seq = inputs["sequence"][:1]
    single_sample_tab = inputs["tabular"][:1]

    batch_seq = inputs["sequence"][:batch_size]
    batch_tab = inputs["tabular"][:batch_size]
    batch_size_actual = len(batch_tab)

    single_latency: Dict[str, Dict[str, float]] = {}
    batch_latency: Dict[str, Dict[str, float]] = {}

    for name in config.MODEL_ORDER:
        model = trained_models[name]
        if name in ["CNN", "RNN"]:
            single_latency[name] = benchmark_model(model, single_sample_seq, model_type="keras")
            batch_latency[name] = benchmark_model(model, batch_seq, model_type="keras")
        elif name == "DFNN":
            single_latency[name] = benchmark_model(model, single_sample_tab, model_type="keras")
            batch_latency[name] = benchmark_model(model, batch_tab, model_type="keras")
        else:
            single_latency[name] = benchmark_model(model, single_sample_tab, model_type="sklearn")
            batch_latency[name] = benchmark_model(model, batch_tab, model_type="sklearn")

    for name, stats in batch_latency.items():
        stats["per_sample_ms"] = float(stats["mean_ms"] / batch_size_actual)

    top_k = min(3, len(config.MODEL_ORDER))
    ensemble_latency = {
        "top_k": top_k,
        "total_mean_ms": float(sum(batch_latency[name]["mean_ms"] for name in config.MODEL_ORDER[:top_k])),
        "per_sample_ms": float(sum(batch_latency[name]["mean_ms"] for name in config.MODEL_ORDER[:top_k]) / batch_size_actual),
    }

    results = {
        "hardware": get_hardware_info(),
        "methodology": {
            "warmup_runs": 10,
            "timed_runs": 100,
            "batch_size": batch_size,
        },
        "single_sample_latency": single_latency,
        "batch_latency": batch_latency,
        "ensemble_latency": ensemble_latency,
    }
    save_json(results, config.RESULTS_DIR / "inference_benchmark.json")
    return results


def run_ensemble(
    seed: int = config.GLOBAL_SEED,
    cv_results: Dict[str, Dict[str, float]] | None = None,
    oof_preds: Dict[str, np.ndarray] | None = None,
    top_k: int | None = None,
    models: List[str] | None = None,
) -> Dict[str, object]:
    """Unified ensemble analysis: correlation, greedy selection, exhaustive search, summary.

    Selection is always OOF-based (no test leakage).  Every candidate subset is
    evaluated on both simple averaging and Ridge stacking on the test set.

    Args:
        top_k:  Override: evaluate a specific top-K (OOF-ranked) subset.
        models: Override: evaluate a specific user-supplied model list.
                (Both overrides appear in the summary table alongside
                 greedy-optimal and exhaustive-best results.)
    """
    log("=" * 60)
    log("STEP 6: ENSEMBLE ANALYSIS")
    log("=" * 60)
    set_seeds(seed)

    X_train_val, y_train_val, X_test, y_test, _, _ = data_loader.load_datasets(feature_set="all_features")

    if oof_preds is None or cv_results is None:
        oof_preds, cv_results = run_cv_experiment(
            feature_set="all_features",
            seed=seed,
            save_results=True,
            use_cache=True,
        )

    _, test_predictions, _, _ = training.train_full_models(
        X_train_val, y_train_val, X_test, feature_set="all_features",
    )

    # ── Metric helpers ──────────────────────────────────────────────────────
    def _avg_metrics(ml: List[str]) -> Dict[str, float]:
        preds = np.mean([test_predictions[name] for name in ml], axis=0)
        return {
            "test_r2": float(r2_score(y_test, preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_mae": float(mean_absolute_error(y_test, preds)),
        }

    def _ridge_metrics(ml: List[str]) -> Dict[str, float]:
        X_oof_sub = np.column_stack([oof_preds[name] for name in ml])
        X_test_sub = np.column_stack([test_predictions[name] for name in ml])
        ridge = ensemble.train_stacking_model(X_oof_sub, y_train_val)
        preds = ridge.predict(X_test_sub)
        return {
            "test_r2": float(r2_score(y_test, preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_mae": float(mean_absolute_error(y_test, preds)),
            "coef": {name: round(float(c), 4) for name, c in zip(ml, ridge.coef_)},
            "intercept": round(float(ridge.intercept_), 4),
        }

    def _eval(ml: List[str], label: str) -> Dict[str, object]:
        return {"method": label, "models": ml, "simple_avg": _avg_metrics(ml), "ridge": _ridge_metrics(ml)}

    # ── 1. Prediction Correlation Matrix (OOF) ─────────────────────────────
    log("\nPrediction Correlation Matrix (OOF):")
    corr, names = ensemble.prediction_correlation_matrix(oof_preds, config.MODEL_ORDER)
    header = f"{'':>10}" + "".join(f"{n:>10}" for n in names)
    log(header)
    for i, row_name in enumerate(names):
        log(f"{row_name:>10}" + "".join(f"{corr[i, j]:>10.3f}" for j in range(len(names))))

    # ── 2. Best single model ────────────────────────────────────────────────
    model_ranking = ensemble.rank_models_by_oof(cv_results)
    best_single = model_ranking[0]
    log(f"\nBest single model (OOF-ranked): {best_single}")

    # ── 3. Greedy Forward Selection (OOF-based, both methods on test) ───────
    log("\n" + "=" * 60)
    log("GREEDY FORWARD SELECTION (OOF-selected, Test-evaluated)")
    log("=" * 60)
    greedy_steps = ensemble.greedy_forward_selection(config.MODEL_ORDER, oof_preds, y_train_val)
    for step in greedy_steps:
        avg_m = _avg_metrics(step["selected_models"])
        rdg_m = _ridge_metrics(step["selected_models"])
        step["test_r2_avg"] = avg_m["test_r2"]
        step["test_r2_ridge"] = rdg_m["test_r2"]

    log(f"  {'Step':<6} {'Added':<12} {'OOF R²':>10} {'Δ OOF':>10} {'Avg R²':>10} {'Ridge R²':>10}")
    log("  " + "-" * 62)
    for step in greedy_steps:
        delta_str = f"{step['delta_r2']:+.4f}" if step["step"] > 1 else "---"
        log(f"  {step['step']:<6} +{step['added_model']:<11} {step['oof_r2']:>10.4f} "
            f"{delta_str:>10} {step['test_r2_avg']:>10.4f} {step['test_r2_ridge']:>10.4f}")

    greedy_optimal = greedy_steps[0]
    for step in greedy_steps[1:]:
        if step["delta_r2"] > 0:
            greedy_optimal = step
        else:
            break
    greedy_models = greedy_optimal["selected_models"]
    log(f"\n  Greedy OOF-optimal (K={len(greedy_models)}): {greedy_models}")

    # ── 4. Exhaustive Subset Search (OOF-ranked, both methods on test) ──────
    log("\n" + "=" * 60)
    log(f"EXHAUSTIVE SUBSET SEARCH ({2 ** len(config.MODEL_ORDER) - 1} subsets)")
    log("=" * 60)
    exhaustive = ensemble.exhaustive_subset_search(config.MODEL_ORDER, oof_preds, y_train_val)
    for r in exhaustive:
        r["test_r2_avg"] = _avg_metrics(r["models"])["test_r2"]
        r["test_r2_ridge"] = _ridge_metrics(r["models"])["test_r2"]

    log(f"  {'Rank':<6} {'K':<4} {'OOF R²':>10} {'Avg R²':>10} {'Ridge R²':>10}  Models")
    log("  " + "-" * 80)
    for rank, r in enumerate(exhaustive[:15], 1):
        log(f"  {rank:<6} {r['k']:<4} {r['oof_r2']:>10.4f} "
            f"{r['test_r2_avg']:>10.4f} {r['test_r2_ridge']:>10.4f}  {r['models']}")

    best_ex_oof   = exhaustive[0]
    best_ex_avg   = max(exhaustive, key=lambda x: x["test_r2_avg"])
    best_ex_ridge = max(exhaustive, key=lambda x: x["test_r2_ridge"])

    log(f"\n  Best by OOF   K={best_ex_oof['k']}: {best_ex_oof['models']}"
        f" → OOF={best_ex_oof['oof_r2']:.4f}, Avg={best_ex_oof['test_r2_avg']:.4f}, Ridge={best_ex_oof['test_r2_ridge']:.4f}")
    log(f"  Best by Avg   K={best_ex_avg['k']}: {best_ex_avg['models']}"
        f" → OOF={best_ex_avg['oof_r2']:.4f}, Avg={best_ex_avg['test_r2_avg']:.4f}, Ridge={best_ex_avg['test_r2_ridge']:.4f}")
    log(f"  Best by Ridge K={best_ex_ridge['k']}: {best_ex_ridge['models']}"
        f" → OOF={best_ex_ridge['oof_r2']:.4f}, Avg={best_ex_ridge['test_r2_avg']:.4f}, Ridge={best_ex_ridge['test_r2_ridge']:.4f}")

    # ── 5. Summary Table ────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("SUMMARY COMPARISON (Test Set)")
    log("=" * 60)

    # build candidate list, deduplicated by model set
    candidates: List[tuple] = [
        (config.MODEL_ORDER, f"All-{len(config.MODEL_ORDER)}"),
        ([best_single],      f"Best single ({best_single})"),
        (greedy_models,      f"Greedy K={len(greedy_models)}: {greedy_models}"),
        (best_ex_oof["models"],   f"Best-OOF   K={best_ex_oof['k']}: {best_ex_oof['models']}"),
        (best_ex_avg["models"],   f"Best-Avg   K={best_ex_avg['k']}: {best_ex_avg['models']}"),
        (best_ex_ridge["models"], f"Best-Ridge K={best_ex_ridge['k']}: {best_ex_ridge['models']}"),
    ]
    if top_k is not None:
        tk_models = model_ranking[:top_k]
        candidates.append((tk_models, f"Top-{top_k} (OOF-ranked): {tk_models}"))
    if models is not None:
        candidates.append((models, f"Manual: {models}"))

    seen: set = set()
    unique: List[tuple] = []
    for ml, lbl in candidates:
        key = tuple(sorted(ml))
        if key not in seen:
            seen.add(key)
            unique.append((ml, lbl))

    log(f"  {'Method':<60} {'Avg R²':>10} {'Ridge R²':>10} {'Avg RMSE':>10} {'Avg MAE':>10}")
    log("  " + "-" * 102)
    summary_rows: List[Dict[str, object]] = []
    for ml, lbl in unique:
        avg_m = _avg_metrics(ml)
        rdg_m = _ridge_metrics(ml)
        log(f"  {lbl:<60} {avg_m['test_r2']:>10.4f} {rdg_m['test_r2']:>10.4f} "
            f"{avg_m['test_rmse']:>10.2f} {avg_m['test_mae']:>10.2f}")
        summary_rows.append({"label": lbl, "models": ml, "simple_avg": avg_m, "ridge": rdg_m})

    log("=" * 60)

    all_results: Dict[str, object] = {
        "correlation_matrix": corr.tolist(),
        "correlation_model_order": names,
        "greedy_steps": greedy_steps,
        "greedy_optimal": _eval(greedy_models, f"Greedy K={len(greedy_models)}"),
        "exhaustive_all": exhaustive,
        "exhaustive_best_oof":   _eval(best_ex_oof["models"],   f"Best by OOF K={best_ex_oof['k']}"),
        "exhaustive_best_avg":   _eval(best_ex_avg["models"],   f"Best by Avg K={best_ex_avg['k']}"),
        "exhaustive_best_ridge": _eval(best_ex_ridge["models"], f"Best by Ridge K={best_ex_ridge['k']}"),
        "summary": summary_rows,
    }
    save_json(all_results, config.RESULTS_DIR / "ensemble_analysis.json")
    return all_results


def run_all(seed: int = config.GLOBAL_SEED, top_k: int | None = None, models: List[str] | None = None):
    log("Running all experiments with caching...")
    oof_preds, cv_results = run_cv_experiment(
        feature_set="all_features",
        seed=seed,
        save_results=True,
        use_cache=True,
    )
    run_top_k_experiment(seed=seed, cv_results=cv_results, oof_preds=oof_preds)
    run_feature_ablation(seed=seed)
    run_drop_one(seed=seed, cv_results=cv_results, oof_preds=oof_preds, top_k=top_k)
    run_ensemble(seed=seed, cv_results=cv_results, oof_preds=oof_preds, top_k=top_k, models=models)
    run_benchmark(seed=seed)
