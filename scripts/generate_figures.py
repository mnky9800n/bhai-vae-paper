#!/usr/bin/env python3
"""Generate all paper figures for BHAI VAE paper."""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys
import os

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler, FEATURE_COLS

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Model paths
UNSUP_MODEL = MODEL_DIR / 'unsup.pt'
SEMISUP_MODEL = MODEL_DIR / 'semisup.pt'
TRAINING_DATA = DATA_DIR / 'vae_training_data_v2_20cm.csv'


def fig_zeroshot_scatter(results_df, save_path=None):
    """Create zero-shot prediction scatter plot."""
    results_df = results_df[
        (results_df['r2_v267'] > -1) & 
        (results_df['r2_v214'] > -1) &
        (results_df['r2_v267'] < 1.1) &
        (results_df['r2_v214'] < 1.1)
    ].copy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = results_df['r2_v214'].values  # Semi-supervised
    y = results_df['r2_v267'].values  # Unsupervised
    n_samples = results_df['n_samples'].values
    
    log_samples = np.log10(np.clip(n_samples, 100, 1e6))
    sizes = 30 + 150 * (log_samples - 2) / 4
    colors = ['#1f77b4' if xi > yi else '#ff7f0e' for xi, yi in zip(x, y)]
    
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=sizes[i], c=colors[i], alpha=0.7, 
                  edgecolors='white', linewidth=0.5)
    
    ax.plot([-0.2, 1.05], [-0.2, 1.05], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Semi-supervised R²', fontsize=14)
    ax.set_ylabel('Unsupervised R²', fontsize=14)
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-0.2, 1.05)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markersize=10, label='Semi-supervised higher'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
               markersize=10, label='Unsupervised higher'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Size legend
    size_legend_ax = fig.add_axes([0.15, 0.72, 0.15, 0.18])
    size_legend_ax.set_xlim(0, 1)
    size_legend_ax.set_ylim(0, 1)
    size_legend_ax.axis('off')
    
    sample_sizes = [100, 1000, 10000, 100000, 1000000]
    sample_labels = ['100', '1k', '10k', '100k', '1M']
    y_positions = [0.85, 0.65, 0.45, 0.25, 0.05]
    
    for ss, label, yp in zip(sample_sizes, sample_labels, y_positions):
        log_s = np.log10(ss)
        size = 30 + 150 * (log_s - 2) / 4
        size_legend_ax.scatter(0.3, yp, s=size, c='gray', alpha=0.7, edgecolors='white')
        size_legend_ax.text(0.6, yp, label, va='center', fontsize=10)
    
    size_legend_ax.text(0.4, 1.0, 'Samples', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return fig


def fig_reconstruction_scatter(X_scaled, pred_unsup, pred_semisup, save_path=None):
    """Create reconstruction scatter plot comparing both models."""
    def r2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    physical_labels = ['Bulk Density', 'Mag. Susc.', 'NGR']
    optical_labels = ['R', 'G', 'B']
    physical_idx = [0, 1, 2]
    optical_idx = [3, 4, 5]
    
    physical_color = '#2c3e50'
    optical_colors = ['#e74c3c', '#27ae60', '#3498db']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Reconstruction Quality: Predicted vs True', fontsize=14, fontweight='bold', y=0.98)
    
    fig.text(0.28, 0.93, 'Physical Properties', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.72, 0.93, 'Optical Properties', ha='center', fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    n_plot = min(10000, len(X_scaled))
    plot_idx = np.random.choice(len(X_scaled), n_plot, replace=False)
    
    for row in range(3):
        for col, (pred, model_name) in enumerate([(pred_unsup, 'v2.6.7'), (pred_semisup, 'v2.14')]):
            ax = axes[row, col]
            feat_idx = physical_idx[row]
            true_vals = X_scaled[plot_idx, feat_idx]
            pred_vals = pred[plot_idx, feat_idx]
            r2_val = r2(X_scaled[:, feat_idx], pred[:, feat_idx])
            
            ax.scatter(true_vals, pred_vals, alpha=0.3, s=1, c=physical_color)
            lims = [min(true_vals.min(), pred_vals.min()) - 0.5,
                    max(true_vals.max(), pred_vals.max()) + 0.5]
            ax.plot(lims, lims, '--', color='gray', alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            ax.text(0.05, 0.95, f'R²={r2_val:.3f}', transform=ax.transAxes, 
                    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if col == 0:
                ax.set_ylabel(f'{physical_labels[row]}\nPredicted', fontsize=10)
            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')
            if row == 2:
                ax.set_xlabel('True', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for col_offset, (pred, model_name) in enumerate([(pred_unsup, 'v2.6.7'), (pred_semisup, 'v2.14')]):
            col = col_offset + 2
            ax = axes[row, col]
            feat_idx = optical_idx[row]
            true_vals = X_scaled[plot_idx, feat_idx]
            pred_vals = pred[plot_idx, feat_idx]
            r2_val = r2(X_scaled[:, feat_idx], pred[:, feat_idx])
            
            ax.scatter(true_vals, pred_vals, alpha=0.3, s=1, c=optical_colors[row])
            lims = [min(true_vals.min(), pred_vals.min()) - 0.5,
                    max(true_vals.max(), pred_vals.max()) + 0.5]
            ax.plot(lims, lims, '--', color='gray', alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            ax.text(0.05, 0.95, f'R²={r2_val:.3f}', transform=ax.transAxes, 
                    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if col == 2:
                ax.set_ylabel(f'{optical_labels[row]}\nPredicted', fontsize=10)
            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')
            if row == 2:
                ax.set_xlabel('True', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return fig


def fig_roc_comparison(emb_unsup, emb_semisup, labels, n_classes, 
                       max_samples=50000, save_path=None):
    """Create fair ROC comparison using same classifier on both embeddings."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from catboost import CatBoostClassifier
    
    if len(labels) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(labels), max_samples, replace=False)
        emb_unsup = emb_unsup[idx]
        emb_semisup = emb_semisup[idx]
        labels = labels[idx]
    
    n = len(labels)
    n_train = int(0.8 * n)
    
    print("Training CatBoost on unsupervised embeddings...")
    clf_unsup = CatBoostClassifier(iterations=500, verbose=False, random_state=42)
    clf_unsup.fit(emb_unsup[:n_train], labels[:n_train])
    
    print("Training CatBoost on semi-supervised embeddings...")
    clf_semisup = CatBoostClassifier(iterations=500, verbose=False, random_state=42)
    clf_semisup.fit(emb_semisup[:n_train], labels[:n_train])
    
    prob_unsup = clf_unsup.predict_proba(emb_unsup[n_train:])
    prob_semisup = clf_semisup.predict_proba(emb_semisup[n_train:])
    
    y_test = labels[n_train:]
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    prob_unsup_full = np.zeros((len(y_test), n_classes))
    prob_semisup_full = np.zeros((len(y_test), n_classes))
    for i, c in enumerate(clf_unsup.classes_):
        prob_unsup_full[:, c] = prob_unsup[:, i]
    for i, c in enumerate(clf_semisup.classes_):
        prob_semisup_full[:, c] = prob_semisup[:, i]
    
    fpr_unsup, tpr_unsup, _ = roc_curve(y_test_bin.ravel(), prob_unsup_full.ravel())
    fpr_semisup, tpr_semisup, _ = roc_curve(y_test_bin.ravel(), prob_semisup_full.ravel())
    
    auc_unsup = auc(fpr_unsup, tpr_unsup)
    auc_semisup = auc(fpr_semisup, tpr_semisup)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr_unsup, tpr_unsup, '#ff7f0e', lw=2.5, 
            label=f'Unsupervised (v2.6.7) AUC = {auc_unsup:.3f}')
    ax.plot(fpr_semisup, tpr_semisup, '#1f77b4', lw=2.5,
            label=f'Semi-supervised (v2.14) AUC = {auc_semisup:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Lithology Classification: ROC Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return fig


def main():
    print("=" * 60)
    print("BHAI VAE Paper Figure Generation")
    print("=" * 60)
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv(TRAINING_DATA)
    X_raw = train_df[FEATURE_COLS].values
    valid_mask = ~np.isnan(X_raw).any(axis=1)
    X_raw = X_raw[valid_mask]
    train_df = train_df[valid_mask].reset_index(drop=True)
    
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"Samples: {len(X_scaled):,}")
    
    # Load models
    print("\nLoading models...")
    model_unsup = VAE(input_dim=6, latent_dim=10)
    model_unsup.load_state_dict(torch.load(UNSUP_MODEL, map_location='cpu'))
    model_unsup.eval()
    
    model_semisup = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=139)
    model_semisup.load_state_dict(torch.load(SEMISUP_MODEL, map_location='cpu'))
    model_semisup.eval()
    print("Models loaded")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    with torch.no_grad():
        X_t = torch.FloatTensor(X_scaled)
        emb_unsup = model_unsup.get_embeddings(X_t).numpy()
        emb_semisup = model_semisup.get_embeddings(X_t).numpy()
    print(f"Embeddings shape: {emb_unsup.shape}")
    
    # Figure 1: Zero-shot scatter (if results exist)
    zeroshot_results_path = DATA_DIR / 'zeroshot_results.csv'
    if zeroshot_results_path.exists():
        print("\n[1/3] Generating zero-shot scatter plot...")
        zeroshot_df = pd.read_csv(zeroshot_results_path)
        fig_zeroshot_scatter(zeroshot_df, OUTPUT_DIR / 'fig_zeroshot_scatter.png')
    else:
        print(f"\n[1/3] Skipping zero-shot scatter (no {zeroshot_results_path})")
    
    # Figure 2: Reconstruction scatter
    print("\n[2/3] Generating reconstruction scatter plot...")
    with torch.no_grad():
        X_t = torch.FloatTensor(X_scaled)
        pred_unsup, _, _ = model_unsup(X_t)
        pred_semisup, _, _, _ = model_semisup(X_t)
        pred_unsup = pred_unsup.numpy()
        pred_semisup = pred_semisup.numpy()
    
    fig_reconstruction_scatter(X_scaled, pred_unsup, pred_semisup, 
                               OUTPUT_DIR / 'fig_reconstruction_scatter.png')
    
    # Figure 3: ROC comparison
    print("\n[3/3] Generating ROC comparison plot...")
    labels_raw = train_df['Principal'].values
    unique_labels = np.unique(labels_raw)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels_raw])
    n_classes = len(unique_labels)
    print(f"Classes: {n_classes}")
    
    fig_roc_comparison(emb_unsup, emb_semisup, y, n_classes,
                       save_path=OUTPUT_DIR / 'fig_roc_comparison.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("Generated figures:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")
    print("=" * 60)


if __name__ == '__main__':
    main()
