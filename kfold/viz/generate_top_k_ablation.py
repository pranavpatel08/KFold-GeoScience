"""
Generate Fig. 2 — Top-K Ablation Performance.

Replicates the original figure style pixel-for-pixel with updated values
from the current well-based GroupKFold results:

  OOF rank order: XGBoost → DFNN → CNN → RNN → RF
  Best ensemble:  K=5 (all 5 models, simple averaging)

Output: kfold/top_k_ablation.png   ← path used by WCCI_IJCNN.tex
"""
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
KFOLD_ROOT = Path(__file__).parent.parent
OUT_PATH   = KFOLD_ROOT / "top_k_ablation.png"   # manuscript reads from here

# ── Data (from results/ensemble_compare_top_k.out) ─────────────────────────
K_SIZES = [1, 2, 3, 4, 5]

# OOF rank order: XGBoost, DFNN, CNN, RNN, RF
MODELS  = ["XGBoost", "+DFNN", "+CNN", "+RNN", "+RF"]
X_LABELS = [f"K={k}\n({m})" for k, m in zip(K_SIZES, MODELS)]

R2    = [0.8202, 0.8779, 0.8868, 0.8881, 0.8905]
RMSE  = [794.53, 654.74, 630.41, 626.67, 619.91]
MAE   = [585.87, 492.15, 472.37, 467.04, 464.31]

RMSE_S = [v / 1000 for v in RMSE]
MAE_S  = [v / 1000 for v in MAE]

# Best ensemble is K=5 (index 4)
BEST = 4

# ── Colors (match original) ─────────────────────────────────────────────────
C_GREEN = "#55a868"   # R²  — muted green
C_BLUE  = "#4c72b0"   # RMSE — muted blue
C_RED   = "#c44e52"   # MAE  — muted red

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Lines
ax.plot(K_SIZES, R2,     "o-",   color=C_GREEN, lw=2.0, markersize=9,  label="R² Score")
ax.plot(K_SIZES, RMSE_S, "s--",  color=C_BLUE,  lw=2.0, markersize=8,  label="RMSE / 1000")
ax.plot(K_SIZES, MAE_S,  "^-.",  color=C_RED,   lw=2.0, markersize=8,  label="MAE / 1000")

# Annotations at best (K=5) — matching original style
ax.annotate(
    f"{R2[BEST]:.4f}",
    xy=(K_SIZES[BEST], R2[BEST]),
    xytext=(0, -14), textcoords="offset points",
    ha="center", va="top", fontsize=10, fontweight="bold", color=C_GREEN,
)
ax.annotate(
    f"{RMSE[BEST]:.0f}",
    xy=(K_SIZES[BEST], RMSE_S[BEST]),
    xytext=(0, -14), textcoords="offset points",
    ha="center", va="top", fontsize=10, fontweight="bold", color=C_BLUE,
)
ax.annotate(
    f"{MAE[BEST]:.0f}",
    xy=(K_SIZES[BEST], MAE_S[BEST]),
    xytext=(0, -14), textcoords="offset points",
    ha="center", va="top", fontsize=10, fontweight="bold", color=C_RED,
)

# Axes
ax.set_xlabel("Number of Models (K)", fontsize=12)
ax.set_ylabel("Scaled Metric Value", fontsize=12)
ax.set_xticks(K_SIZES)
ax.set_xticklabels(X_LABELS)
ax.set_ylim(0.4, 1.0)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Grid — light, both axes
ax.grid(True, axis="both", color="lightgray", linewidth=0.8, alpha=0.5)

# Title
ax.set_title("Top-K Ablation Performance", pad=12)

# Legend — upper-right inside, with frame (matches original position)
ax.legend(loc="upper right", frameon=True, framealpha=0.9,
          edgecolor="lightgray", bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
plt.close()
