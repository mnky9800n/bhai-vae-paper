#!/usr/bin/env python3
"""
Train VAE models (unsupervised or semi-supervised).

Usage:
    python train_vae.py --model unsupervised --output models/unsup.pt
    python train_vae.py --model semisupervised --output models/semisup.pt --alpha 0.1
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler, FEATURE_COLS


def load_data(data_path, lithology_col='Principal'):
    """Load and preprocess training data."""
    df = pd.read_csv(data_path)
    
    # Extract features
    X = df[FEATURE_COLS].values
    
    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    
    # Get lithology labels if present
    labels = None
    label_map = None
    if lithology_col in df.columns:
        labels_raw = df[lithology_col].values[valid_mask]
        unique_labels = np.unique(labels_raw)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels_raw])
    
    # Scale features
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, labels, scaler, label_map


def train_unsupervised(X, epochs=100, batch_size=256, lr=1e-3, beta=1.0, 
                       device='cpu', verbose=True):
    """Train unsupervised VAE."""
    model = VAE(input_dim=6, latent_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = {'loss': [], 'recon': [], 'kl': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = model.loss_function(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        epoch_loss /= len(X)
        epoch_recon /= len(X)
        epoch_kl /= len(X)
        
        history['loss'].append(epoch_loss)
        history['recon'].append(epoch_recon)
        history['kl'].append(epoch_kl)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, "
                  f"Recon={epoch_recon:.4f}, KL={epoch_kl:.4f}", flush=True)
    
    return model, history


def train_semisupervised(X, labels, epochs=100, batch_size=256, lr=1e-3, 
                        beta=1.0, alpha=0.1, n_classes=139, device='cpu', verbose=True):
    """Train semi-supervised VAE."""
    model = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = {'loss': [], 'recon': [], 'kl': [], 'class': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        epoch_class = 0
        
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar, logits = model(x)
            loss, recon_loss, kl_loss, class_loss = model.loss_function(
                recon, x, mu, logvar, logits, y, beta, alpha
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_class += class_loss.item()
        
        epoch_loss /= len(X)
        epoch_recon /= len(X)
        epoch_kl /= len(X)
        epoch_class /= len(X)
        
        history['loss'].append(epoch_loss)
        history['recon'].append(epoch_recon)
        history['kl'].append(epoch_kl)
        history['class'].append(epoch_class)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, "
                  f"Recon={epoch_recon:.4f}, KL={epoch_kl:.4f}, Class={epoch_class:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train VAE models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['unsupervised', 'semisupervised'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, default='data/vae_training_data_v2_20cm.csv',
                       help='Path to training data')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Classification loss weight (semi-supervised only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Training {args.model} VAE")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Beta: {args.beta}")
    if args.model == 'semisupervised':
        print(f"  Alpha: {args.alpha}")
    print()
    
    # Load data
    X, labels, scaler, label_map = load_data(args.data)
    print(f"Loaded {len(X):,} samples")
    
    if labels is not None:
        n_classes = len(label_map)
        print(f"Found {n_classes} lithology classes")
    
    # Train
    if args.model == 'unsupervised':
        model, history = train_unsupervised(
            X, epochs=args.epochs, batch_size=args.batch_size, 
            lr=args.lr, beta=args.beta, device=args.device
        )
    else:
        if labels is None:
            raise ValueError("Semi-supervised training requires lithology labels")
        model, history = train_semisupervised(
            X, labels, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, beta=args.beta, alpha=args.alpha, 
            n_classes=n_classes, device=args.device
        )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to {output_path}")
    
    # Save history
    history_path = output_path.with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Saved history to {history_path}")


if __name__ == '__main__':
    main()
