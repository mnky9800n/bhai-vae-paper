#!/usr/bin/env python3
"""Remake reconstruction scatter without subsampling."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(BASE_DIR))
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler, FEATURE_COLS

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / "vae_training_data_v2_20cm.csv")
feature_cols = FEATURE_COLS
lith_col = 'Principal'

mask = df[feature_cols].notna().all(axis=1) & df[lith_col].notna()
df_clean = df[mask].copy()

lith_encoder = LabelEncoder()
y = lith_encoder.fit_transform(df_clean[lith_col])
n_classes = len(lith_encoder.classes_)

X = df_clean[feature_cols].values.astype(np.float32)
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)

# Load models
print("Loading models...")
model_unsup = VAE(input_dim=6, latent_dim=10, hidden_dims=[64, 32])
model_unsup.load_state_dict(torch.load(MODEL_DIR / "unsup.pt", map_location=DEVICE))
model_unsup.to(DEVICE)
model_unsup.eval()

model_semisup = SemiSupervisedVAE(input_dim=6, latent_dim=10, hidden_dims=[64, 32], n_classes=n_classes)
model_semisup.load_state_dict(torch.load(MODEL_DIR / "semisup.pt", map_location=DEVICE))
model_semisup.to(DEVICE)
model_semisup.eval()

# Get reconstructions
print("Getting reconstructions...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
    try:
        recon_unsup, _, _, _ = model_unsup(X_tensor)
    except:
        recon_unsup, _, _ = model_unsup(X_tensor)
    try:
        recon_semisup, _, _, _ = model_semisup(X_tensor)
    except:
        recon_semisup, _, _ = model_semisup(X_tensor)
    recon_unsup = recon_unsup.cpu().numpy()
    recon_semisup = recon_semisup.cpu().numpy()

# Inverse transform
X_orig = scaler.inverse_transform(X_scaled)
X_unsup = scaler.inverse_transform(recon_unsup)
X_semisup = scaler.inverse_transform(recon_semisup)

# Print per-variable R² 
print("\nPer-variable R² scores:")
feature_names = ['Bulk Density', 'Mag. Susc.', 'NGR', 'R', 'G', 'B']
for i, name in enumerate(feature_names):
    r2_u = r2_score(X_orig[:, i], X_unsup[:, i])
    r2_s = r2_score(X_orig[:, i], X_semisup[:, i])
    print(f"  {name:15s}  unsup={r2_u:.4f}  semisup={r2_s:.4f}  delta={r2_s-r2_u:+.4f}")

r2_all_u = r2_score(X_scaled.flatten(), recon_unsup.flatten())
r2_all_s = r2_score(X_scaled.flatten(), recon_semisup.flatten())
print(f"\n  Overall (scaled): unsup={r2_all_u:.4f}  semisup={r2_all_s:.4f}")

# Plot WITHOUT subsampling
print("\nGenerating plot (no subsampling)...")
fig, axes = plt.subplots(2, 6, figsize=(18, 8))

for i, name in enumerate(feature_names):
    # Unsupervised (top row)
    ax = axes[0, i]
    r2 = r2_score(X_orig[:, i], X_unsup[:, i])
    ax.scatter(X_orig[:, i], X_unsup[:, i], alpha=0.05, s=0.5, rasterized=True)
    ax.plot([X_orig[:, i].min(), X_orig[:, i].max()],
            [X_orig[:, i].min(), X_orig[:, i].max()], 'r--', lw=2)
    ax.set_xlabel(f'True {name}')
    ax.set_ylabel(f'Predicted {name}')
    ax.set_title(f'Unsup R²={r2:.3f}')
    
    # Semi-supervised (bottom row)
    ax = axes[1, i]
    r2 = r2_score(X_orig[:, i], X_semisup[:, i])
    ax.scatter(X_orig[:, i], X_semisup[:, i], alpha=0.05, s=0.5, rasterized=True)
    ax.plot([X_orig[:, i].min(), X_orig[:, i].max()],
            [X_orig[:, i].min(), X_orig[:, i].max()], 'r--', lw=2)
    ax.set_xlabel(f'True {name}')
    ax.set_ylabel(f'Predicted {name}')
    ax.set_title(f'Semi-sup R²={r2:.3f}')

axes[0, 0].set_ylabel('Unsupervised\nPredicted', fontsize=12)
axes[1, 0].set_ylabel('Semi-supervised\nPredicted', fontsize=12)

plt.tight_layout()
out_path = FIGURES_DIR / 'fig_reconstruction_scatter.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out_path}")
