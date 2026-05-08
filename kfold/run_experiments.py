"""CLI for GeoMeta Ensemble experiments."""
from __future__ import annotations

import argparse

import experiments
from utils import set_seeds


def main():
    parser = argparse.ArgumentParser(description="GeoMeta Ensemble Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["cv", "top_k", "features", "drop_one", "ensemble", "benchmark", "all"],
        help="Which experiment to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Fixed number of top models to use")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models for ensemble (e.g. XGBoost,CNN,RNN,RF,DFNN)",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    models_list = [m.strip() for m in args.models.split(",")] if args.models else None

    if args.experiment == "cv":
        experiments.run_cv_experiment(seed=args.seed)
    elif args.experiment == "top_k":
        experiments.run_top_k_experiment(seed=args.seed)
    elif args.experiment == "features":
        experiments.run_feature_ablation(seed=args.seed)
    elif args.experiment == "drop_one":
        experiments.run_drop_one(seed=args.seed, top_k=args.top_k)
    elif args.experiment == "ensemble":
        experiments.run_ensemble(seed=args.seed, top_k=args.top_k, models=models_list)
    elif args.experiment == "benchmark":
        experiments.run_benchmark(seed=args.seed)
    elif args.experiment == "all":
        experiments.run_all(seed=args.seed, top_k=args.top_k, models=models_list)


if __name__ == "__main__":
    main()
