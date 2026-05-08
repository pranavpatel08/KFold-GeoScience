"""Advanced visualizations for KFold stacking results.

These plots reveal patterns NOT easily seen in summary tables:
- Depth-dependent model behavior
- Pressure regime performance
- Model complementarity
- Calibration quality
- Error distributions
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import r2_score
from scipy import stats

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Paths - viz lives at kfold/viz/, repo root holds data/ and kfold/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
KFOLD_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = KFOLD_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models" / "all_features"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALL_FEATURES = [
    "tvd", "dt", "dt_nct", "gr", "sphi", "hp", "ob",
    "rhob_combined", "res_deep", "eaton_ratio",
    "hp_gradient", "ob_gradient", "tvd_normalized"
]
MODEL_ORDER = ["XGBoost", "DFNN", "CNN", "RNN", "RF"]
STABLE_MODELS = ["XGBoost", "DFNN", "CNN", "RNN", "RF"]
COLORS = {
    "CNN": "#e41a1c", "DFNN": "#ff7f00", "RNN": "#377eb8",
    "RF": "#4daf4a", "XGBoost": "#984ea3", "Ensemble": "#252525"
}


def load_test_data():
    """Load test data with metadata."""
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    X = test_df[ALL_FEATURES].values.astype(np.float32)
    y = test_df["ppp"].values.astype(np.float32)
    well_id = test_df["well_id"].values
    tvd = test_df["tvd"].values
    hp = test_df["hp"].values  # Hydrostatic pressure for regime classification
    return X, y, well_id, tvd, hp


def load_models_and_predict(X):
    """Load cached models and generate predictions."""
    x_scaler = joblib.load(MODELS_DIR / "x_scaler.joblib")
    y_scaler = joblib.load(MODELS_DIR / "y_scaler.joblib")
    dfnn_scaler = joblib.load(MODELS_DIR / "dfnn_scaler.joblib")
    
    X_s = x_scaler.transform(X)
    X_cnn = X_s.reshape(-1, X.shape[1], 1)
    X_dfnn = dfnn_scaler.transform(X)
    
    predictions = {}
    
    cnn = tf.keras.models.load_model(MODELS_DIR / "cnn.keras")
    preds = cnn.predict(X_cnn, verbose=0).flatten()
    predictions["CNN"] = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    
    dfnn = tf.keras.models.load_model(MODELS_DIR / "dfnn.keras")
    preds = dfnn.predict(X_dfnn, verbose=0).flatten()
    predictions["DFNN"] = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    
    rnn = tf.keras.models.load_model(MODELS_DIR / "rnn.keras")
    preds = rnn.predict(X_cnn, verbose=0).flatten()
    predictions["RNN"] = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    
    rf = joblib.load(MODELS_DIR / "rf.joblib")
    predictions["RF"] = rf.predict(X_s)
    
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    predictions["XGBoost"] = xgb.predict(X_s)
    
    # Simple average of all 5 models
    predictions["Ensemble"] = np.mean([
        predictions["XGBoost"], 
        predictions["DFNN"], 
        predictions["CNN"], 
        predictions["RNN"], 
        predictions["RF"]
    ], axis=0)
    
    return predictions


def classify_pressure_regime(ppp, hp):
    """Classify samples into pressure regimes."""
    ratio = ppp / hp
    regimes = np.full(len(ppp), "Normal", dtype=object)
    regimes[ratio < 0.95] = "Underpressured"
    regimes[ratio > 1.05] = "Overpressured"
    return regimes


# =============================================================================
# PLOT 1: Pressure Profile Along Depth (Well Log Style)
# =============================================================================
def plot_pressure_profile(predictions, y_true, tvd, well_id, save_path=None):
    """
    Well log style plot showing predicted vs actual pressure along depth.
    Reveals: depth-dependent prediction quality, transition zone behavior.
    """
    selected_well = "PINDORI-2"
    
    mask = well_id == selected_well
    tvd_w = tvd[mask]
    y_w = y_true[mask]
    
    # Sort by depth
    sort_idx = np.argsort(tvd_w)
    tvd_w = tvd_w[sort_idx]
    y_w = y_w[sort_idx]
    
    # Publication-ready figure size (fits IEEE full page width)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 9), sharey=True)
    
    # Panel 1: Actual vs Ensemble
    ax1 = axes[0]
    ens_w = predictions["Ensemble"][mask][sort_idx]
    ax1.plot(y_w, tvd_w, 'k-', lw=1.5, label='Actual', alpha=0.9)
    ax1.plot(ens_w, tvd_w, 'r-', lw=1.2, label='Ensemble', alpha=0.8)
    ax1.fill_betweenx(tvd_w, y_w, ens_w, alpha=0.3, color='red', label='Error')
    ax1.set_xlabel('Pore Pressure (psi)', fontsize=12)
    ax1.set_ylabel('TVD (m)', fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.invert_yaxis()
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Ensemble Prediction', fontsize=11, fontweight='semibold')
    
    # Panel 2: RF vs XGBoost (tree models)
    ax2 = axes[1]
    rf_w = predictions["RF"][mask][sort_idx]
    xgb_w = predictions["XGBoost"][mask][sort_idx]
    ax2.plot(y_w, tvd_w, 'k-', lw=1.5, label='Actual', alpha=0.9)
    ax2.plot(rf_w, tvd_w, '-', color=COLORS["RF"], lw=1.2, label='RF', alpha=0.8)
    ax2.plot(xgb_w, tvd_w, '-', color=COLORS["XGBoost"], lw=1.2, label='XGBoost', alpha=0.8)
    ax2.set_xlabel('Pore Pressure (psi)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Tree Models', fontsize=11, fontweight='semibold')
    
    # Panel 3: CNN vs RNN (neural networks)
    ax3 = axes[2]
    cnn_w = predictions["CNN"][mask][sort_idx]
    rnn_w = predictions["RNN"][mask][sort_idx]
    ax3.plot(y_w, tvd_w, 'k-', lw=1.5, label='Actual', alpha=0.9)
    ax3.plot(cnn_w, tvd_w, '-', color=COLORS["CNN"], lw=1.2, label='CNN', alpha=0.8)
    ax3.plot(rnn_w, tvd_w, '-', color=COLORS["RNN"], lw=1.2, label='RNN', alpha=0.8)
    ax3.set_xlabel('Pore Pressure (psi)', fontsize=12)
    ax3.tick_params(axis='both', labelsize=10)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_title('Neural Networks', fontsize=11, fontweight='semibold')
    
    fig.suptitle(f'Pressure Profile: {selected_well}', fontsize=13, fontweight='semibold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 2: Error Distribution by Pressure Regime
# =============================================================================
def plot_error_by_regime(predictions, y_true, hp, save_path=None):
    """
    Violin plots of prediction error by pressure regime.
    Reveals: which pressure conditions are hardest to predict.
    """
    regimes = classify_pressure_regime(y_true, hp)
    
    # Prepare data for violin plot
    data = []
    for model in STABLE_MODELS + ["Ensemble"]:
        errors = y_true - predictions[model]
        for regime, err in zip(regimes, errors):
            data.append({"Model": model, "Regime": regime, "Error (psi)": err})
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Order regimes logically
    regime_order = ["Underpressured", "Normal", "Overpressured"]
    
    sns.violinplot(
        data=df, x="Model", y="Error (psi)", hue="Regime",
        hue_order=regime_order, palette="Set2", inner="quartile",
        ax=ax, cut=0, scale="width"
    )
    
    ax.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.7)
    ax.set_title("Prediction Error Distribution by Pressure Regime")
    ax.set_ylabel("Error (Actual − Predicted, psi)")
    ax.legend(title="Regime", loc="upper right")
    
    # Add sample counts
    regime_counts = pd.Series(regimes).value_counts()
    count_str = "  |  ".join([f"{r}: {regime_counts[r]:,}" for r in regime_order])
    ax.text(0.5, -0.12, f"Sample counts: {count_str}", transform=ax.transAxes,
            ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 3: Cumulative Error Distribution (CDF)
# =============================================================================
def plot_error_cdf(predictions, y_true, save_path=None):
    """
    CDF of absolute errors for each model.
    Reveals: What fraction of predictions fall within X psi error.
    More informative than single MAE/RMSE values.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for model in STABLE_MODELS + ["Ensemble"]:
        abs_errors = np.abs(y_true - predictions[model])
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        ax.plot(sorted_errors, cdf * 100, label=model, color=COLORS[model], lw=1.5)
    
    # Reference lines for practical thresholds
    for thresh in [100, 250, 500]:
        ax.axvline(thresh, color='gray', linestyle=':', lw=0.8, alpha=0.5)
        ax.text(thresh + 10, 5, f'{thresh}', fontsize=8, color='gray')
    
    ax.set_xlabel('Absolute Error (psi)')
    ax.set_ylabel('Cumulative % of Predictions')
    ax.set_title('Cumulative Error Distribution')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add percentile annotations for Ensemble
    ens_errors = np.abs(y_true - predictions["Ensemble"])
    p50, p90, p95 = np.percentile(ens_errors, [50, 90, 95])
    ax.text(0.02, 0.98, f"Ensemble:\n  50%: ≤{p50:.0f} psi\n  90%: ≤{p90:.0f} psi\n  95%: ≤{p95:.0f} psi",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 4: Model Complementarity (When models disagree)
# =============================================================================
def plot_model_complementarity(predictions, y_true, save_path=None):
    """
    Scatter plot: RF error vs RNN error (the two best diverse models).
    Reveals: Whether models make independent errors (good for ensemble).
    Points in different quadrants = complementary errors.
    """
    rf_err = y_true - predictions["RF"]
    rnn_err = y_true - predictions["RNN"]
    cnn_err = y_true - predictions["CNN"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # RF vs RNN
    ax1 = axes[0]
    ax1.scatter(rf_err, rnn_err, alpha=0.1, s=5, c='#2c7fb8', edgecolors='none')
    ax1.axhline(0, color='gray', linestyle='-', lw=0.5)
    ax1.axvline(0, color='gray', linestyle='-', lw=0.5)
    ax1.set_xlabel('RF Error (psi)')
    ax1.set_ylabel('RNN Error (psi)')
    ax1.set_title('RF vs RNN Errors')
    
    # Add quadrant labels
    lim = 2000
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.text(lim*0.7, lim*0.7, 'Both\nunder-predict', ha='center', fontsize=8, color='gray')
    ax1.text(-lim*0.7, -lim*0.7, 'Both\nover-predict', ha='center', fontsize=8, color='gray')
    ax1.text(lim*0.7, -lim*0.7, 'Complementary\nerrors', ha='center', fontsize=8, color='green')
    ax1.text(-lim*0.7, lim*0.7, 'Complementary\nerrors', ha='center', fontsize=8, color='green')
    
    # Correlation
    corr = np.corrcoef(rf_err, rnn_err)[0, 1]
    ax1.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax1.transAxes, va='top', fontsize=10)
    
    # RF vs CNN
    ax2 = axes[1]
    ax2.scatter(rf_err, cnn_err, alpha=0.1, s=5, c='#e41a1c', edgecolors='none')
    ax2.axhline(0, color='gray', linestyle='-', lw=0.5)
    ax2.axvline(0, color='gray', linestyle='-', lw=0.5)
    ax2.set_xlabel('RF Error (psi)')
    ax2.set_ylabel('CNN Error (psi)')
    ax2.set_title('RF vs CNN Errors')
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    
    corr2 = np.corrcoef(rf_err, cnn_err)[0, 1]
    ax2.text(0.02, 0.98, f'r = {corr2:.3f}', transform=ax2.transAxes, va='top', fontsize=10)
    
    fig.suptitle('Model Error Complementarity (lower correlation = better ensemble diversity)', fontsize=11, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 5: Depth-Stratified R² Heatmap
# =============================================================================
def plot_depth_stratified_r2(predictions, y_true, tvd, save_path=None):
    """
    Heatmap of R² by depth bin for each model.
    Transposed format: Depth on Y-axis (vertical), Models on X-axis.
    Fits single-column IEEE format better.
    """
    # Create depth bins
    n_bins = 10
    depth_bins = pd.qcut(tvd, n_bins, labels=False, duplicates='drop')
    bin_edges = np.percentile(tvd, np.linspace(0, 100, n_bins + 1))
    # Labels for Y-axis (Depth intervals)
    bin_labels = [f'{int(bin_edges[i])}–{int(bin_edges[i+1])}m' for i in range(len(bin_edges)-1)]
    
    # Compute R² for each model in each depth bin
    models = STABLE_MODELS + ["Ensemble"]
    r2_matrix = np.zeros((n_bins, len(models))) # Transposed: (Depth, Models)
    
    for j in range(n_bins):
        mask = depth_bins == j
        for i, model in enumerate(models):
            if mask.sum() > 10:  # Need enough samples
                r2 = r2_score(y_true[mask], predictions[model][mask])
                # Clip negative R2 for visualization clarity (optional, but requested better readability)
                # Let's keep raw values but color-code appropriately
                r2_matrix[j, i] = r2
            else:
                r2_matrix[j, i] = np.nan
    
    # Single column width approx 3.5 inches. 
    # Tall aspect ratio to match well log depth orientation.
    fig, ax = plt.subplots(figsize=(3.5, 6))
    
    # Plot heatmap
    # cmap: RdYlGn (Red=Bad, Green=Good)
    im = ax.imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
    
    # X-axis: Models
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Model', fontsize=10)
    
    # Y-axis: Depth Bins
    ax.set_yticks(np.arange(n_bins))
    ax.set_yticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Depth Interval (TVD)', fontsize=10)
    
    # Invert Y-axis so shallow is top, deep is bottom (standard well log view)
    # Note: imshow by default puts index 0 at top, which matches our bin order (shallow->deep)
    # So we don't strictly need ax.invert_yaxis() if bin 0 is shallowest.
    # Checked: bin_edges is sorted, so index 0 = shallowest. 
    
    ax.set_title('R² by Depth Interval', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for j in range(n_bins):  # Rows (Depth)
        for i in range(len(models)):  # Cols (Models)
            val = r2_matrix[j, i]
            if not np.isnan(val):
                # Adaptive text color
                color = 'white' if (val < 0.4 or val > 0.8) else 'black'
                
                # Format text: simpler if <0
                text_val = f'{val:.2f}' if val > -10 else '<-10'
                
                ax.text(i, j, text_val, ha='center', va='center', fontsize=8, color=color)
    
    # Colorbar at the bottom
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.12, shrink=0.9)
    cbar.set_label('R² Score', fontsize=9)
    cbar.ax.tick_params(labelsize=8) 
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 6: Calibration Plot (Reliability Diagram)
# =============================================================================
def plot_calibration(predictions, y_true, save_path=None):
    """
    Binned prediction vs actual mean.
    Reveals: Whether predictions are well-calibrated (points on diagonal = good).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    models_to_plot = ["RF", "RNN", "Ensemble"]
    n_bins = 20
    
    for model in models_to_plot:
        preds = predictions[model]
        
        # Bin predictions
        bin_edges = np.percentile(preds, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        actual_means = []
        
        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(preds[mask].mean())
                actual_means.append(y_true[mask].mean())
        
        ax.plot(bin_centers, actual_means, 'o-', label=model, color=COLORS[model], 
                markersize=5, lw=1.5, alpha=0.8)
    
    # Perfect calibration line
    lims = [y_true.min(), y_true.max()]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.7, label='Perfect')
    
    ax.set_xlabel('Predicted Pore Pressure (psi)')
    ax.set_ylabel('Actual Mean Pore Pressure (psi)')
    ax.set_title('Calibration Plot (Reliability Diagram)')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 7: Feature Importance Comparison
# =============================================================================
def plot_feature_importance(save_path=None):
    """
    Side-by-side RF vs XGBoost feature importances.
    Reveals: Which features drive predictions for each model type.
    """
    rf = joblib.load(MODELS_DIR / "rf.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    
    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_
    
    # Normalize
    rf_imp = rf_imp / rf_imp.sum() * 100
    xgb_imp = xgb_imp / xgb_imp.sum() * 100
    
    # Sort by average importance
    avg_imp = (rf_imp + xgb_imp) / 2
    sort_idx = np.argsort(avg_imp)[::-1]
    
    features_sorted = [ALL_FEATURES[i] for i in sort_idx]
    rf_sorted = rf_imp[sort_idx]
    xgb_sorted = xgb_imp[sort_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(features_sorted))
    width = 0.35
    
    ax.barh(x - width/2, rf_sorted, width, label='Random Forest', color=COLORS["RF"], alpha=0.8)
    ax.barh(x + width/2, xgb_sorted, width, label='XGBoost', color=COLORS["XGBoost"], alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Feature Importance: RF vs XGBoost')
    ax.legend(loc='lower right')
    
    # Highlight top 3
    for i in range(3):
        ax.get_yticklabels()[i].set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# PLOT 8: Per-Well R² Comparison
# =============================================================================
def plot_per_well_r2(predictions, y_true, well_id, save_path=None):
    """
    Grouped bar chart of R² for each test well by model.
    Reveals: Which wells are hardest; which models generalize best.
    """
    wells = np.unique(well_id)
    models = STABLE_MODELS + ["Ensemble"]
    
    r2_data = {model: [] for model in models}
    
    for well in wells:
        mask = well_id == well
        for model in models:
            r2_data[model].append(r2_score(y_true[mask], predictions[model][mask]))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(wells))
    width = 0.15
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, r2_data[model], width, label=model, color=COLORS[model], alpha=0.85)
    
    ax.set_xlabel('Test Well')
    ax.set_ylabel('R² Score')
    ax.set_title('Per-Well Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(wells, rotation=20, ha='right', fontsize=9)
    ax.legend(loc='lower left', ncol=3)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def main():
    print("Loading test data...")
    X, y, well_id, tvd, hp = load_test_data()
    
    print("Loading models and generating predictions...")
    predictions = load_models_and_predict(X)
    
    print("\nGenerating visualizations...")
    
    # 1. Pressure Profile (Well Log Style)
    plot_pressure_profile(
        predictions, y, tvd, well_id,
        save_path=FIGURES_DIR / "pressure_profile.png"
    )
    
    # 2. Error by Pressure Regime
    plot_error_by_regime(
        predictions, y, hp,
        save_path=FIGURES_DIR / "error_by_regime.png"
    )
    
    # 3. Cumulative Error Distribution
    plot_error_cdf(
        predictions, y,
        save_path=FIGURES_DIR / "error_cdf.png"
    )
    
    # 4. Model Complementarity
    plot_model_complementarity(
        predictions, y,
        save_path=FIGURES_DIR / "model_complementarity.png"
    )
    
    # 5. Depth-Stratified R²
    plot_depth_stratified_r2(
        predictions, y, tvd,
        save_path=FIGURES_DIR / "depth_r2_heatmap.png"
    )
    
    # 6. Calibration Plot
    plot_calibration(
        predictions, y,
        save_path=FIGURES_DIR / "calibration_plot.png"
    )
    
    # 7. Feature Importance
    plot_feature_importance(
        save_path=FIGURES_DIR / "feature_importance.png"
    )
    
    # 8. Per-Well R²
    plot_per_well_r2(
        predictions, y, well_id,
        save_path=FIGURES_DIR / "per_well_r2.png"
    )
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
