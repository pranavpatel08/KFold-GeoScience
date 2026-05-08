# K-Fold Stacking for Pore Pressure Prediction

Reference implementation for **"Simple Averaging vs. Learned Stacking in K-Fold Ensembles: Evidence from Pore Pressure Prediction"** — Patel, Amjad, Varghese, & Amjad, IEEE IJCNN 2026 (to appear).

A K-Fold stacking framework using well-based `GroupKFold` cross-validation, where ensemble composition is selected by **out-of-fold (OOF) R²** — never by test-set performance. We evaluate five architecturally distinct base learners (CNN, DFNN, RNN, Random Forest, XGBoost) on 186,070 samples from 17 development wells and report results on 4 fully blind test wells (84,904 samples) from Pakistan's Potwar Basin.

## Authors

- **Pranav Patel** — Khoury College of Computer Sciences, Northeastern University
- **Muhammad Raiees Amjad** — Department of Computer Science, Bahria University *(corresponding author for data requests)*
- **Rohan Benjamin Varghese** — Khoury College of Computer Sciences, Northeastern University
- **Tehmina Amjad** — Khoury College of Computer Sciences, Northeastern University

## Paper

[`paper/Patel_IJCNN2026_kfold-stacking-pore-pressure.pdf`](paper/Patel_IJCNN2026_kfold-stacking-pore-pressure.pdf) — camera-ready, to appear at IEEE IJCNN 2026 (WCCI).

## Headline result

The 5-model simple-average ensemble achieves **R² = 0.8905** on 4 blind test wells (RMSE = 619.91 psi, MAE = 464.31 psi), outperforming the best single model (XGBoost, R² = 0.8202) by **+7.03 percentage points**. Simple averaging consistently beats Ridge meta-learner stacking — and the gap *widens* as the ensemble grows (from 0.41 pp at K=2 to 2.36 pp at K=5). Under cross-well distribution shift, equal weighting preserves architectural diversity that learned weights discard.

## Repository layout

```
KFold-GeoScience/
├── README.md                                          # this file
├── LICENSE                                            # MIT
├── CITATION.cff                                       # IJCNN 2026 (to appear)
├── requirements.txt
├── paper/
│   └── Patel_IJCNN2026_kfold-stacking-pore-pressure.pdf
├── preprocessing/
│   ├── preprocessing_pipeline.py                     # raw LAS→CSV pipeline
│   └── preprocessing_metadata.json                    # well splits + feature config
├── kfold/
│   ├── config.py                                      # paths, hyperparameters, feature lists
│   ├── data_loader.py                                 # CSV loading + well-id grouping
│   ├── models.py                                      # CNN / DFNN / RNN / RF / XGBoost builders
│   ├── training.py                                    # OOF generation, full-data retraining
│   ├── ensemble.py                                    # ranking, top-K, drop-one, greedy/exhaustive search
│   ├── experiments.py                                 # experiment runners (cached)
│   ├── utils.py                                       # logging, seeding, hardware info
│   ├── run_experiments.py                             # CLI entry point
│   └── viz/                                           # plotting (requires data + cached models)
└── data/                                              # users place CSVs here (not shipped)
```

## Installation

Python 3.12+ required. NVIDIA GPU recommended (paper used a single H200; an H100/A100/RTX 40-series will work).

```bash
git clone https://github.com/pranavpatel08/KFold-GeoScience.git
cd KFold-GeoScience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The well-log data from Pakistan's Potwar Basin are **available for research purposes upon request to the corresponding author** (M. R. Amjad, Bahria University).

Once you have data, place three CSV files at `data/`:

- `data/train_data.csv` — 13 development wells used for fold construction
- `data/val_data.csv` — 4 development wells; merged into the development set in `data_loader.py`
- `data/test_data.csv` — 4 blind test wells (RAJIAN-03A, PINDORI-2, TURKWAL DEEP X 2, Balkassar POL 01)

Each CSV must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `tvd` | float | True vertical depth (m) |
| `dt` | float | Compressional travel time (μs/ft) |
| `dt_nct` | float | Normal compaction trend |
| `gr` | float | Gamma ray (API) |
| `sphi` | float | Sonic porosity |
| `hp` | float | Hydrostatic pressure (psi) |
| `ob` | float | Overburden stress (psi) |
| `rhob_combined` | float | Bulk density (g/cc) |
| `res_deep` | float | Deep resistivity (Ωm) |
| `eaton_ratio` | float | (dt/dt_nct)³ |
| `hp_gradient` | float | hp / (tvd × 3.28084) |
| `ob_gradient` | float | ob / (tvd × 3.28084) |
| `tvd_normalized` | float | tvd / max(tvd) |
| `ppp` | float | Pore pressure target (psi) |
| `well_id` | str | Well identifier (used for `GroupKFold`) |

`preprocessing/preprocessing_pipeline.py` is the canonical pipeline — given a directory of per-well LAS-derived CSVs, it applies physical-constraint filtering, engineers the four physics features above, and writes the three split CSVs plus `preprocessing_metadata.json`. Run it with `python preprocessing/preprocessing_pipeline.py` after placing per-well CSVs in `./datasets/`.

## Quick start

After installing dependencies and placing CSVs in `data/`, run experiments with:

```bash
python kfold/run_experiments.py --experiment cv         # OOF generation (Table II)
python kfold/run_experiments.py --experiment top_k      # Top-K ablation (Fig. 2 / Table III)
python kfold/run_experiments.py --experiment ensemble   # Simple-avg vs Ridge across K (Section IV-C)
python kfold/run_experiments.py --experiment features   # Feature ablation
python kfold/run_experiments.py --experiment drop_one   # Drop-one analysis
python kfold/run_experiments.py --experiment benchmark  # Inference latency
python kfold/run_experiments.py --experiment all        # Run everything
```

Outputs land in `kfold/results/` (CV results, OOF predictions, ablation CSVs, cached models). The `viz/` scripts plot from those cached outputs and require `--experiment all` to have been run first.

## Method

The pipeline runs in three stages:

1. **OOF prediction generation** — 5-fold `GroupKFold` on the 17 development wells, with `well_id` as the grouping variable. No well appears in both training and validation folds of any split. Per fold: `StandardScaler` is fit on training folds only, each base model is trained, and predictions are generated for the held-out fold. Concatenating across folds yields out-of-fold predictions for every development sample.
2. **Model ranking and combination** — base models are ranked by OOF R² (never by test performance). Both simple averaging and a Ridge regression meta-learner (α = 1.0) trained on OOF predictions are evaluated as combination strategies.
3. **Final evaluation** — base models are retrained on the full development set and applied to the 4 blind test wells. Top-K ablation incrementally adds models in OOF rank order to measure how performance scales with ensemble size.

Cross-well GroupKFold is essential here: sample-level splits inflate R² to 0.99+ but do not generalize to unseen wells.

## Results

### OOF performance (5-fold well-based GroupKFold; Table II)

| Model | OOF R² | RMSE (psi) | MAE (psi) |
|---|---|---|---|
| XGBoost | **0.8537** | 683.09 | 515.49 |
| DFNN | 0.8475 | 697.38 | 536.97 |
| CNN | 0.8400 | 714.45 | 538.08 |
| RNN | 0.7905 | 817.41 | 592.09 |
| RF | 0.7859 | 826.49 | 619.09 |

All five models achieve OOF R² between 0.79 and 0.85. The compressed range reflects genuine cross-well generalization difficulty — no model memorizes well-specific patterns.

### Simple averaging vs Ridge stacking on the blind test set (Table III)

| K | Method | R² | RMSE (psi) | MAE (psi) |
|---|---|---|---|---|
| 1 | XGBoost (best single) | 0.8202 | 794.53 | 585.87 |
| 2 | Simple avg | 0.8779 | 654.74 | 492.15 |
| 2 | Ridge | 0.8738 | 665.59 | 507.37 |
| 3 | Simple avg | 0.8868 | 630.41 | 472.37 |
| 3 | Ridge | 0.8819 | 643.94 | 492.83 |
| 4 | Simple avg | 0.8881 | 626.67 | 467.04 |
| 4 | Ridge | 0.8800 | 649.18 | 498.46 |
| 5 | **Simple avg** | **0.8905** | **619.91** | **464.31** |
| 5 | Ridge | 0.8669 | 683.53 | 518.43 |

At K=5, Ridge assigns coefficients XGBoost=0.759, CNN=0.346, DFNN=0.309, RNN=−0.086, RF=−0.200 — actively suppressing two of the five models. Simple averaging preserves their complementary error patterns and wins by 2.36 pp.

## Citation

Camera-ready (to appear at IEEE IJCNN 2026 / WCCI):

```bibtex
@inproceedings{patel2026kfold,
  title     = {Simple Averaging vs.\ Learned Stacking in K-Fold Ensembles:
               Evidence from Pore Pressure Prediction},
  author    = {Patel, Pranav and Amjad, Muhammad Raiees and
               Varghese, Rohan Benjamin and Amjad, Tehmina},
  booktitle = {Proc.\ IEEE Int.\ Joint Conf.\ on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {to appear}
}
```

## License

[MIT](LICENSE).

## Acknowledgments

AI-assisted tools (Claude, ChatGPT) supported code development and manuscript drafting. All experimental design, implementation, and interpretation were conducted by the authors.
