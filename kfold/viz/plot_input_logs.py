"""
Script to visualize input well logs for a representative well.
Generates a publication-quality log plot.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
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
    "axes.spines.top": True,   # Log plots often have ticks on top
    "axes.spines.right": True,
})

# Paths - viz lives at kfold/viz/, repo root holds data/ and kfold/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
KFOLD_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = KFOLD_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_well_logs():
    data_path = DATA_DIR / "test_data.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # Select PINDORI-2
    target_well = "PINDORI-2"
    well_df = df[df["well_id"] == target_well].copy()
    
    if well_df.empty:
        print(f"Well {target_well} not found. Using most sampled well.")
        target_well = df["well_id"].value_counts().idxmax()
        well_df = df[df["well_id"] == target_well].copy()
        
    # ZOOM IN: Focus on 3000-3800m where the interesting transition happens
    # This prevents the 40k points from looking like solid blocks in a printed paper
    mask = (well_df["tvd"] >= 3000) & (well_df["tvd"] <= 3800)
    if mask.sum() > 1000:
        well_df = well_df[mask].copy()
        print(f"Zooming to 3000-3800m interval ({len(well_df)} samples) for clarity.")
    
    print(f"Plotting logs for: {target_well} ({len(well_df)} samples)")
    
    # Sort by depth
    well_df = well_df.sort_values("tvd")
    depth = well_df["tvd"].values
    
    # Plot configuration - Full Page Width for Double Column (approx 7.2 inches)
    # Aspect ratio tall enough to show logs clearly
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 10), sharey=True)
    
    # Adjust whitespace between tracks
    plt.subplots_adjust(wspace=0.15)
    
    # Track 1: Gamma Ray
    ax1 = axes[0]
    ax1.plot(well_df["gr"], depth, color='green', lw=1.0)
    ax1.set_xlim(0, 75) 
    ax1.set_xlabel("Gamma Ray\n(API)", fontsize=10)
    ax1.set_title("Lithology", fontsize=11, fontweight='semibold')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.fill_betweenx(depth, well_df["gr"], 0, color='green', alpha=0.1)

    # Track 2: Resistivity (Log Scale)
    ax2 = axes[1]
    ax2.semilogx(well_df["res_deep"], depth, color='red', lw=1.0)
    ax2.set_xlim(0.2, 2000) # Standard resistivity scale
    ax2.set_xlabel("Resistivity\n(ohm.m)", fontsize=10)
    ax2.set_title("Resistivity", fontsize=11, fontweight='semibold')
    ax2.grid(True, which='both', alpha=0.3)

    # Track 3: Porosity & Density
    ax3 = axes[2]
    # SPHI (Porosity) - Scaled 0.45 to -0.15 (reversed, standard log practice)
    # Note: We plot density here as primary. 
    ax3.plot(well_df["rhob_combined"], depth, color='black', lw=1.0, label="RHOB")
    
    ax3.set_xlim(1.95, 2.95)
    ax3.set_xlabel("Density\n(g/cc)", fontsize=10)
    ax3.set_title("Density", fontsize=11, fontweight='semibold')
    ax3.grid(True, which='both', alpha=0.3)

    # Track 4: Sonic & NCT (Overpressure)
    ax4 = axes[3]
    ax4.plot(well_df["dt"], depth, color='blue', lw=1.0, label="DT")
    if "dt_nct" in well_df.columns:
        ax4.plot(well_df["dt_nct"], depth, color='gray', linestyle='--', lw=1.5, label="NCT")
    
    ax4.set_xlim(40, 140) # Standard DT scale
    ax4.set_xlabel("DT\n(us/ft)", fontsize=10)
    ax4.set_title("Sonic / NCT", fontsize=11, fontweight='semibold')
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax4.grid(True, which='both', alpha=0.3)

    # Common Y-axis
    axes[0].set_ylabel("Depth (TVD m)", fontsize=11)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].invert_yaxis() # Depth increases downwards
    
    # Add title for the zoomed section
    plt.suptitle(f"Petrophysical Well Logs: {target_well} (3000-3800m)", fontsize=12, fontweight='semibold', y=0.96)
    
    # Tight layout but with rect to make room for suptitle
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    
    save_path = OUTPUT_DIR / "input_well_log.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_well_logs()
