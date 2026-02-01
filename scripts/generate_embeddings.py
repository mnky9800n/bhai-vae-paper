#!/usr/bin/env python3
"""
Generate embeddings from trained VAE models and attach to data.

Creates a unified table with all variables and embeddings from both models.

Usage:
    python generate_embeddings.py \
        --unsup-model models/unsup.pt \
        --semisup-model models/semisup.pt \
        --data data/vae_training_data_v2_20cm.csv \
        --output data/embeddings.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler, FEATURE_COLS


def load_model(model_path, model_type='unsupervised'):
    """Load a trained model from checkpoint."""
    if model_type == 'unsupervised':
        model = VAE(input_dim=6, latent_dim=10)
    else:
        model = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=139)
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from VAE models')
    parser.add_argument('--unsup-model', type=str, required=True,
                       help='Path to unsupervised model')
    parser.add_argument('--semisup-model', type=str, required=True,
                       help='Path to semi-supervised model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV with embeddings')
    parser.add_argument('--latent-dim', type=int, default=10,
                       help='Latent dimension')
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = pd.read_csv(args.data)
    X = df[FEATURE_COLS].values
    
    # Handle missing values
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Valid rows: {valid_mask.sum():,}")
    
    # Scale features
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # Load models
    print("\nLoading models...")
    model_unsup = load_model(args.unsup_model, 'unsupervised')
    model_semisup = load_model(args.semisup_model, 'semisupervised')
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    with torch.no_grad():
        X_t = torch.FloatTensor(X_scaled)
        emb_unsup = model_unsup.get_embeddings(X_t).numpy()
        emb_semisup = model_semisup.get_embeddings(X_t).numpy()
    
    print(f"  Unsupervised embeddings: {emb_unsup.shape}")
    print(f"  Semi-supervised embeddings: {emb_semisup.shape}")
    
    # Create output dataframe
    print("\nCreating output table...")
    
    # Start with valid rows from original data
    output_df = df[valid_mask].copy().reset_index(drop=True)
    
    # Add unsupervised embeddings
    for i in range(args.latent_dim):
        output_df[f'unsup_emb_{i}'] = emb_unsup[:, i]
    
    # Add semi-supervised embeddings
    for i in range(args.latent_dim):
        output_df[f'semisup_emb_{i}'] = emb_semisup[:, i]
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"\nSaved embeddings to {output_path}")
    print(f"  Columns: {len(output_df.columns)}")
    print(f"  Rows: {len(output_df):,}")
    
    # Print column summary
    print("\nColumn summary:")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Unsupervised embeddings: unsup_emb_0 ... unsup_emb_{args.latent_dim-1}")
    print(f"  Semi-supervised embeddings: semisup_emb_0 ... semisup_emb_{args.latent_dim-1}")


if __name__ == '__main__':
    main()
