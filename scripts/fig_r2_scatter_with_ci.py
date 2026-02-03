#!/usr/bin/env python3
"""
Generate R² scatter plot comparing unsupervised vs semi-supervised VAE
with 95% confidence intervals.

Usage:
    uv run python scripts/fig_r2_scatter_with_ci.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"


def compute_r2_standard_error(r2: float, n: int) -> float:
    """
    Approximate standard error of R² using analytical formula.
    
    SE(R²) ≈ sqrt((1-R²)² × 4×R² / n)
    
    This is an approximation valid for moderate to large n.
    For more accurate CIs, use bootstrap resampling.
    """
    r2_clip = np.clip(r2, 0.01, 0.99)
    return np.sqrt((1 - r2_clip)**2 * 4 * r2_clip / n)


def add_confidence_intervals(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns to results dataframe."""
    df = results_df.copy()
    
    df['r2_v267_se'] = df.apply(
        lambda row: compute_r2_standard_error(row['r2_v267'], row['n_samples']), axis=1
    )
    df['r2_v214_se'] = df.apply(
        lambda row: compute_r2_standard_error(row['r2_v214'], row['n_samples']), axis=1
    )
    
    # 95% CI = mean ± 1.96 * SE
    df['r2_v267_ci_low'] = df['r2_v267'] - 1.96 * df['r2_v267_se']
    df['r2_v267_ci_high'] = df['r2_v267'] + 1.96 * df['r2_v267_se']
    df['r2_v214_ci_low'] = df['r2_v214'] - 1.96 * df['r2_v214_se']
    df['r2_v214_ci_high'] = df['r2_v214'] + 1.96 * df['r2_v214_se']
    
    return df


def create_scatter_plot(results_df: pd.DataFrame, save_path: Path = None):
    """
    Create scatter plot comparing R² scores with confidence intervals.
    
    - X-axis: Semi-supervised R² (v2.14)
    - Y-axis: Unsupervised R² (v2.6.7)
    - Color: Cyan if semi-supervised better, Pink if unsupervised better
    - Size: Scaled by log(n_samples)
    - Error bars: 95% CI
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = results_df['r2_v214'].values  # Semi-supervised
    y = results_df['r2_v267'].values  # Unsupervised
    xerr = results_df['r2_v214_se'].values * 1.96  # 95% CI
    yerr = results_df['r2_v267_se'].values * 1.96
    n_samples = results_df['n_samples'].values
    
    # Scale point size by log of sample count
    log_samples = np.log10(np.clip(n_samples, 100, 1e6))
    sizes = 30 + 150 * (log_samples - 2) / 4
    
    # Color: Cyan if semi-supervised wins, Pink if unsupervised wins
    colors = ['#00BCD4' if xi > yi else '#E91E63' for xi, yi in zip(x, y)]
    
    # Count wins
    semisup_wins = sum(1 for xi, yi in zip(x, y) if xi > yi)
    unsup_wins = sum(1 for xi, yi in zip(x, y) if yi > xi)
    ties = len(x) - semisup_wins - unsup_wins
    
    print(f"Semi-supervised wins: {semisup_wins}")
    print(f"Unsupervised wins: {unsup_wins}")
    print(f"Ties: {ties}")
    
    # Error bars (lighter color, behind points)
    for i in range(len(x)):
        ax.errorbar(x[i], y[i], xerr=xerr[i], yerr=yerr[i], 
                    fmt='none', ecolor=colors[i], alpha=0.3, capsize=0, zorder=1)
    
    # Scatter points on top
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=sizes[i], c=colors[i], alpha=0.7, 
                   edgecolors='white', linewidth=0.5, zorder=2)
    
    # Diagonal line (y = x)
    ax.plot([-0.2, 1.05], [-0.2, 1.05], 'k--', lw=1.5, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Semi-supervised R²', fontsize=14)
    ax.set_ylabel('Unsupervised R²', fontsize=14)
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-0.2, 1.05)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend for colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00BCD4', 
               markersize=10, label=f'Semi-supervised better ({semisup_wins})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63', 
               markersize=10, label=f'Unsupervised better ({unsup_wins})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Size legend
    size_legend_ax = fig.add_axes([0.15, 0.72, 0.15, 0.18])
    size_legend_ax.set_xlim(0, 1)
    size_legend_ax.set_ylim(0, 1)
    size_legend_ax.axis('off')
    
    for ss, label, yp in [(100, '100', 0.85), (1000, '1k', 0.65), 
                           (10000, '10k', 0.45), (100000, '100k', 0.25)]:
        log_s = np.log10(ss)
        size = 30 + 150 * (log_s - 2) / 4
        size_legend_ax.scatter(0.3, yp, s=size, c='gray', alpha=0.7, edgecolors='white')
        size_legend_ax.text(0.6, yp, label, va='center', fontsize=10)
    size_legend_ax.text(0.4, 1.0, 'n', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return fig


def main():
    """Load results, add CIs, and generate scatter plot."""
    # Load zeroshot results
    results_path = DATA_DIR / "zeroshot_scatter_results_full.csv"
    
    if not results_path.exists():
        # Try alternate location
        results_path = Path("/home/mnky9800n/clawd/bhai-analysis/zeroshot_scatter_results_full.csv")
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"Loading results from: {results_path}")
    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} variables")
    
    # Add confidence intervals
    results_df = add_confidence_intervals(results_df)
    
    # Save results with CIs
    ci_path = DATA_DIR / "zeroshot_scatter_results_with_ci.csv"
    results_df.to_csv(ci_path, index=False)
    print(f"Saved results with CIs: {ci_path}")
    
    # Generate plot
    fig_path = FIGURES_DIR / "fig_r2_unsup_vs_semi.png"
    create_scatter_plot(results_df, save_path=fig_path)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Mean R² (semi-supervised): {results_df['r2_v214'].mean():.3f}")
    print(f"Mean R² (unsupervised): {results_df['r2_v267'].mean():.3f}")
    print(f"Median R² (semi-supervised): {results_df['r2_v214'].median():.3f}")
    print(f"Median R² (unsupervised): {results_df['r2_v267'].median():.3f}")


if __name__ == "__main__":
    main()
