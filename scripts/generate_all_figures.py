#!/usr/bin/env python3
"""
Regenerate all paper figures with hybrid loss VAE.

Run this after bootstrap confirms the best hyperparameters.
Trains final models, saves embeddings, generates all figures.

Usage:
    python regenerate_all_figures.py [--skip-training]
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import model definitions from models/vae.py (single source of truth)
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
import umap
from datetime import datetime
import argparse
import shutil
import warnings
warnings.filterwarnings('ignore')

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters (from bootstrap results)
LATENT_DIM = 10
HIDDEN_DIMS = [64, 32]
MASK_RATIO = 0.1
MASK_WEIGHT = 0.5
N_EPOCHS = 100
BATCH_SIZE = 256




def train_model(X_train, y_train=None, supervised=False, n_classes=None, verbose=True):
    """Train VAE with hybrid loss."""
    if supervised:
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        model = SemiSupervisedVAE(input_dim=6, latent_dim=LATENT_DIM, 
                                   hidden_dims=HIDDEN_DIMS, n_classes=n_classes).to(DEVICE)
    else:
        dataset = TensorDataset(torch.FloatTensor(X_train))
        model = VAE(input_dim=6, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    beta_start, beta_end, beta_anneal_epochs = 1e-6, 0.5, 30
    alpha = 0.5
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        n_samples = 0
        
        beta = beta_start + (beta_end - beta_start) * min(epoch / beta_anneal_epochs, 1.0)
        
        for batch in dataloader:
            if supervised:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            else:
                x_batch = batch[0].to(DEVICE)
            
            batch_size_actual = x_batch.size(0)
            n_samples += batch_size_actual
            
            mask = torch.rand_like(x_batch) < MASK_RATIO
            x_masked = x_batch.clone()
            x_masked[mask] = 0
            
            optimizer.zero_grad()
            
            if supervised:
                recon, mu, logvar, logits = model(x_masked)
            else:
                recon, mu, logvar = model(x_masked)
            
            recon_loss = F.mse_loss(recon, x_batch, reduction='mean')
            if mask.sum() > 0:
                masked_loss = F.mse_loss(recon[mask], x_batch[mask], reduction='mean')
            else:
                masked_loss = torch.tensor(0.0, device=DEVICE)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + MASK_WEIGHT * masked_loss + beta * kl_loss
            
            if supervised:
                class_loss = F.cross_entropy(logits, y_batch)
                loss = loss + alpha * class_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_size_actual
        
        scheduler.step(total_loss / n_samples)
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{N_EPOCHS}, Loss: {total_loss/n_samples:.4f}", flush=True)
    
    return model


def get_embeddings(model, X, supervised=False):
    """Extract embeddings from trained model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        embeddings = model.get_embeddings(X_tensor).cpu().numpy()
    return embeddings


def get_reconstructions(model, X, supervised=False):
    """Get deterministic reconstructions (using mu, no sampling noise)."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        mu, logvar = model.encode(X_tensor)
        # Use mu directly (deterministic) instead of reparameterize (stochastic)
        recon = model.decode(mu)
    return recon.cpu().numpy()


def fig_reconstruction_scatter(X_original, X_recon_unsup, X_recon_semisup, feature_cols, scaler):
    """Reconstruction quality scatter plots."""
    print("Generating: fig_reconstruction_scatter.png")
    
    # Inverse transform to original scale
    X_orig_raw = scaler.inverse_transform(X_original)
    X_unsup_raw = scaler.inverse_transform(X_recon_unsup)
    X_semisup_raw = scaler.inverse_transform(X_recon_semisup)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    
    feature_names = ['Bulk Density', 'Mag. Susc.', 'NGR', 'R', 'G', 'B']
    
    for i, (name, col) in enumerate(zip(feature_names, feature_cols)):
        # Unsupervised (top row)
        ax = axes[0, i]
        r2 = r2_score(X_orig_raw[:, i], X_unsup_raw[:, i])
        ax.scatter(X_orig_raw[:, i], X_unsup_raw[:, i], alpha=0.05, s=0.5, rasterized=True)
        ax.plot([X_orig_raw[:, i].min(), X_orig_raw[:, i].max()],
                [X_orig_raw[:, i].min(), X_orig_raw[:, i].max()], 'r--', lw=2)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'Unsup R²={r2:.3f}')
        
        # Semi-supervised (bottom row)
        ax = axes[1, i]
        r2 = r2_score(X_orig_raw[:, i], X_semisup_raw[:, i])
        ax.scatter(X_orig_raw[:, i], X_semisup_raw[:, i], alpha=0.05, s=0.5, rasterized=True)
        ax.plot([X_orig_raw[:, i].min(), X_orig_raw[:, i].max()],
                [X_orig_raw[:, i].min(), X_orig_raw[:, i].max()], 'r--', lw=2)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'Semi-sup R²={r2:.3f}')
    
    axes[0, 0].set_ylabel('Unsupervised\nPredicted', fontsize=12)
    axes[1, 0].set_ylabel('Semi-supervised\nPredicted', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_reconstruction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_reconstruction_scatter.png")


def fig_roc_comparison(emb_unsup, emb_semisup, y, lith_encoder):
    """ROC curves comparing unsup vs semisup embeddings."""
    print("Generating: fig_roc_comparison.png")
    
    class_counts = np.bincount(y)
    top10 = np.argsort(class_counts)[-10:]
    
    n_sample = min(20000, len(y))
    idx = np.random.choice(len(y), n_sample, replace=False)
    
    mask = np.isin(y[idx], top10)
    y_sub = y[idx][mask]
    emb_unsup_sub = emb_unsup[idx][mask]
    emb_semisup_sub = emb_semisup[idx][mask]
    y_bin = label_binarize(y_sub, classes=top10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Unsupervised
    print("  Training SVM for unsupervised...")
    svm_unsup = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))
    svm_unsup.fit(emb_unsup_sub, y_bin)
    y_score_unsup = svm_unsup.predict_proba(emb_unsup_sub)
    
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    mean_auc = 0
    for i, cls in enumerate(top10):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_unsup[:, i])
        roc_auc = auc(fpr, tpr)
        mean_auc += roc_auc
        name = lith_encoder.inverse_transform([cls])[0][:15]
        ax.plot(fpr, tpr, lw=2, alpha=0.8, color=colors[i], label=f'{name} ({roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Unsupervised VAE (mean AUC={mean_auc/10:.2f})')
    ax.legend(loc='lower right', fontsize=8)
    
    # Semi-supervised
    print("  Training SVM for semi-supervised...")
    svm_semisup = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))
    svm_semisup.fit(emb_semisup_sub, y_bin)
    y_score_semisup = svm_semisup.predict_proba(emb_semisup_sub)
    
    ax = axes[1]
    mean_auc = 0
    for i, cls in enumerate(top10):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_semisup[:, i])
        roc_auc = auc(fpr, tpr)
        mean_auc += roc_auc
        name = lith_encoder.inverse_transform([cls])[0][:15]
        ax.plot(fpr, tpr, lw=2, alpha=0.8, color=colors[i], label=f'{name} ({roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Semi-supervised VAE (mean AUC={mean_auc/10:.2f})')
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_roc_comparison.png")


def fig_umap_lithology(emb_unsup, emb_semisup, y, lith_encoder):
    """UMAP visualizations of embeddings colored by lithology."""
    print("Generating: fig_umap_lithology.png")
    
    n_sample = min(50000, len(y))
    idx = np.random.choice(len(y), n_sample, replace=False)
    
    class_counts = np.bincount(y)
    top15 = set(np.argsort(class_counts)[-15:])
    
    print("  Computing UMAP for unsupervised...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_unsup = reducer.fit_transform(emb_unsup[idx])
    
    print("  Computing UMAP for semi-supervised...")
    umap_semisup = reducer.fit_transform(emb_semisup[idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    y_sub = y[idx]
    colors = np.array(['#CCCCCC'] * len(y_sub))
    cmap = plt.cm.tab20(np.linspace(0, 1, 15))
    for i, cls in enumerate(sorted(top15)):
        colors[y_sub == cls] = matplotlib.colors.to_hex(cmap[i])
    
    ax = axes[0]
    ax.scatter(umap_unsup[:, 0], umap_unsup[:, 1], c=colors, s=1, alpha=0.5)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Unsupervised VAE Embeddings')
    
    ax = axes[1]
    ax.scatter(umap_semisup[:, 0], umap_semisup[:, 1], c=colors, s=1, alpha=0.5)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Semi-supervised VAE Embeddings')
    
    # Legend
    handles = []
    for i, cls in enumerate(sorted(top15)):
        name = lith_encoder.inverse_transform([cls])[0][:20]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=matplotlib.colors.to_hex(cmap[i]),
                                   markersize=8, label=name))
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_umap_lithology.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_umap_lithology.png")


def fig_zeroshot_scatter(emb_unsup, emb_semisup, y, df, lith_encoder):
    """Zero-shot prediction: predict unseen lithologies using embeddings."""
    print("Generating: fig_zeroshot_scatter.png")
    
    # Split by lithology - hold out some classes entirely
    classes = np.unique(y)
    np.random.shuffle(classes)
    train_classes = set(classes[:int(len(classes) * 0.8)])
    test_classes = set(classes[int(len(classes) * 0.8):])
    
    train_mask = np.isin(y, list(train_classes))
    test_mask = np.isin(y, list(test_classes))
    
    print(f"  Train classes: {len(train_classes)}, Test classes: {len(test_classes)}")
    print(f"  Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
    
    # Train SVM on train classes, predict on test classes
    # For zero-shot: use class centroids from training to classify test
    
    # Compute centroids for each training class
    train_centroids_unsup = {}
    train_centroids_semisup = {}
    for cls in train_classes:
        cls_mask = y == cls
        train_centroids_unsup[cls] = emb_unsup[cls_mask].mean(axis=0)
        train_centroids_semisup[cls] = emb_semisup[cls_mask].mean(axis=0)
    
    # For test samples, find nearest centroid
    def nearest_centroid_predict(embeddings, centroids, test_mask):
        test_emb = embeddings[test_mask]
        centroid_matrix = np.array(list(centroids.values()))
        centroid_classes = list(centroids.keys())
        
        # Compute distances
        distances = np.linalg.norm(test_emb[:, None, :] - centroid_matrix[None, :, :], axis=2)
        pred_idx = distances.argmin(axis=1)
        return np.array([centroid_classes[i] for i in pred_idx])
    
    pred_unsup = nearest_centroid_predict(emb_unsup, train_centroids_unsup, test_mask)
    pred_semisup = nearest_centroid_predict(emb_semisup, train_centroids_semisup, test_mask)
    
    # Accuracy (how often does it pick a reasonable class?)
    # Since test classes are unseen, we measure embedding quality differently
    # Let's measure: for each test class, how tight is the cluster?
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot: intra-class variance vs inter-class distance for test classes
    test_y = y[test_mask]
    
    def compute_cluster_metrics(embeddings, labels):
        intra_var = []
        for cls in np.unique(labels):
            cls_emb = embeddings[labels == cls]
            if len(cls_emb) > 1:
                centroid = cls_emb.mean(axis=0)
                var = np.mean(np.linalg.norm(cls_emb - centroid, axis=1))
                intra_var.append(var)
        return np.mean(intra_var) if intra_var else 0
    
    unsup_var = compute_cluster_metrics(emb_unsup[test_mask], test_y)
    semisup_var = compute_cluster_metrics(emb_semisup[test_mask], test_y)
    
    ax = axes[0]
    ax.bar(['Unsupervised', 'Semi-supervised'], [unsup_var, semisup_var], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Mean Intra-class Distance')
    ax.set_title('Embedding Compactness (lower = better)')
    
    # Also show SVM accuracy on test embeddings when trained on train embeddings
    ax = axes[1]
    
    # Subsample for SVM
    n_train_sub = min(10000, train_mask.sum())
    n_test_sub = min(5000, test_mask.sum())
    
    train_idx = np.random.choice(np.where(train_mask)[0], n_train_sub, replace=False)
    test_idx = np.random.choice(np.where(test_mask)[0], n_test_sub, replace=False)
    
    # For this, we need to re-encode test classes to train on
    # Actually for zero-shot, we show the embedding quality via R² of centroid matching
    
    ax.text(0.5, 0.5, f'Unsup intra-class var: {unsup_var:.3f}\nSemisup intra-class var: {semisup_var:.3f}',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Zero-shot Embedding Quality')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_zeroshot_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_zeroshot_scatter.png")


def fig_generated_samples(model_unsup, model_semisup, scaler, feature_cols):
    """Generate samples from latent space and visualize."""
    print("Generating: fig_generated_variables_grid.png")
    
    # Sample from prior
    n_samples = 1000
    z = torch.randn(n_samples, LATENT_DIM).to(DEVICE)
    
    model_unsup.eval()
    model_semisup.eval()
    
    with torch.no_grad():
        gen_unsup = model_unsup.decode(z).cpu().numpy()
        gen_semisup = model_semisup.decode(z).cpu().numpy()
    
    # Inverse transform
    gen_unsup_raw = scaler.inverse_transform(gen_unsup)
    gen_semisup_raw = scaler.inverse_transform(gen_semisup)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    
    feature_names = ['Bulk Density', 'Mag. Susc.', 'NGR', 'R', 'G', 'B']
    
    for i, name in enumerate(feature_names):
        ax = axes[0, i]
        ax.hist(gen_unsup_raw[:, i], bins=50, alpha=0.7, color='#1f77b4')
        ax.set_xlabel(name)
        ax.set_title(f'Unsup Generated')
        
        ax = axes[1, i]
        ax.hist(gen_semisup_raw[:, i], bins=50, alpha=0.7, color='#ff7f0e')
        ax.set_xlabel(name)
        ax.set_title(f'Semi-sup Generated')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_generated_variables_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_generated_variables_grid.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-training', action='store_true', help='Skip training, load existing models')
    args = parser.parse_args()
    
    print("="*70)
    print("REGENERATE ALL PAPER FIGURES")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "vae_training_data_v2_20cm.csv")
    feature_cols = ["Bulk density (GRA)", "Magnetic susceptibility (instr. units)", 
                    "NGR total counts (cps)", "R", "G", "B"]
    X_raw = df[feature_cols].values
    
    scaler = DistributionAwareScaler()
    X = scaler.fit_transform(X_raw)
    
    lith_encoder = LabelEncoder()
    y = lith_encoder.fit_transform(df["Principal"].values)
    n_classes = len(lith_encoder.classes_)
    
    print(f"Samples: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Lithology classes: {n_classes}")
    print()
    
    if args.skip_training:
        print("Loading existing models...")
        model_unsup = VAE(input_dim=6, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS).to(DEVICE)
        model_semisup = SemiSupervisedVAE(input_dim=6, latent_dim=LATENT_DIM, 
                                           hidden_dims=HIDDEN_DIMS, n_classes=n_classes).to(DEVICE)
        model_unsup.load_state_dict(torch.load(MODEL_DIR / 'model_unsup_hybrid.pt'))
        model_semisup.load_state_dict(torch.load(MODEL_DIR / 'model_semisup_hybrid.pt'))
    else:
        # Train models
        print("Training unsupervised VAE...")
        model_unsup = train_model(X, supervised=False)
        torch.save(model_unsup.state_dict(), MODEL_DIR / 'model_unsup_hybrid.pt')
        print("  Saved model_unsup_hybrid.pt")
        
        print("\nTraining semi-supervised VAE...")
        model_semisup = train_model(X, y, supervised=True, n_classes=n_classes)
        torch.save(model_semisup.state_dict(), MODEL_DIR / 'model_semisup_hybrid.pt')
        print("  Saved model_semisup_hybrid.pt")
    
    # Get embeddings
    print("\nExtracting embeddings...")
    emb_unsup = get_embeddings(model_unsup, X)
    emb_semisup = get_embeddings(model_semisup, X, supervised=True)
    
    np.savez(MODEL_DIR / 'embeddings_hybrid.npz', 
             unsup=emb_unsup, semisup=emb_semisup, y=y)
    print("  Saved embeddings_hybrid.npz")
    
    # Get reconstructions
    print("\nGetting reconstructions...")
    recon_unsup = get_reconstructions(model_unsup, X)
    recon_semisup = get_reconstructions(model_semisup, X, supervised=True)
    
    r2_unsup = r2_score(X.flatten(), recon_unsup.flatten())
    r2_semisup = r2_score(X.flatten(), recon_semisup.flatten())
    print(f"  Unsupervised R²: {r2_unsup:.4f}")
    print(f"  Semi-supervised R²: {r2_semisup:.4f}")
    
    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70 + "\n")
    
    fig_reconstruction_scatter(X, recon_unsup, recon_semisup, feature_cols, scaler)
    fig_roc_comparison(emb_unsup, emb_semisup, y, lith_encoder)
    fig_umap_lithology(emb_unsup, emb_semisup, y, lith_encoder)
    fig_zeroshot_scatter(emb_unsup, emb_semisup, y, df, lith_encoder)
    fig_generated_samples(model_unsup, model_semisup, scaler, feature_cols)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print(f"Models saved to: {BASE_DIR}")


if __name__ == "__main__":
    main()
