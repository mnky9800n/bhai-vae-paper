#!/usr/bin/env python3
"""
Run SVM classification on VAE embeddings.

Compares classification performance using embeddings from
unsupervised vs semi-supervised VAE.

Usage:
    python run_svm.py --embeddings data/embeddings.csv --output results/svm_results.json
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def get_embedding_columns(df, prefix):
    """Get columns matching embedding prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description='Run SVM on VAE embeddings')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output results JSON')
    parser.add_argument('--lithology-col', type=str, default='Principal',
                       help='Column name for lithology labels')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Max samples to use (for speed)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM kernel')
    parser.add_argument('--C', type=float, default=1.0,
                       help='SVM regularization parameter')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SVM CLASSIFICATION ON VAE EMBEDDINGS")
    print("=" * 60)
    
    # Load data
    print("\nLoading embeddings...")
    df = pd.read_csv(args.embeddings)
    print(f"  Loaded {len(df):,} samples")
    
    # Get embedding columns
    unsup_cols = get_embedding_columns(df, 'unsup_emb_')
    semisup_cols = get_embedding_columns(df, 'semisup_emb_')
    
    print(f"  Unsupervised embedding dims: {len(unsup_cols)}")
    print(f"  Semi-supervised embedding dims: {len(semisup_cols)}")
    
    # Get labels
    if args.lithology_col not in df.columns:
        raise ValueError(f"Lithology column '{args.lithology_col}' not found")
    
    labels = df[args.lithology_col].values
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])
    
    print(f"  Lithology classes: {n_classes}")
    
    # Subsample if needed
    if len(df) > args.max_samples:
        print(f"\nSubsampling to {args.max_samples:,} samples...")
        np.random.seed(args.seed)
        idx = np.random.choice(len(df), args.max_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        y = y[idx]
    
    # Extract embeddings
    X_unsup = df[unsup_cols].values
    X_semisup = df[semisup_cols].values
    
    # Train/test split (no stratify to avoid single-sample class issues)
    print("\nSplitting data...")
    X_unsup_train, X_unsup_test, y_train, y_test = train_test_split(
        X_unsup, y, test_size=args.test_size, random_state=args.seed
    )
    X_semisup_train, X_semisup_test, _, _ = train_test_split(
        X_semisup, y, test_size=args.test_size, random_state=args.seed
    )
    
    print(f"  Train: {len(y_train):,}")
    print(f"  Test: {len(y_test):,}")
    
    # Scale embeddings
    scaler_unsup = StandardScaler()
    X_unsup_train_scaled = scaler_unsup.fit_transform(X_unsup_train)
    X_unsup_test_scaled = scaler_unsup.transform(X_unsup_test)
    
    scaler_semisup = StandardScaler()
    X_semisup_train_scaled = scaler_semisup.fit_transform(X_semisup_train)
    X_semisup_test_scaled = scaler_semisup.transform(X_semisup_test)
    
    results = {}
    
    # Train SVM on unsupervised embeddings
    print("\n" + "=" * 40)
    print("Training SVM on UNSUPERVISED embeddings...")
    print("=" * 40)
    
    svm_unsup = SVC(kernel=args.kernel, C=args.C, probability=True, random_state=args.seed)
    svm_unsup.fit(X_unsup_train_scaled, y_train)
    
    y_pred_unsup = svm_unsup.predict(X_unsup_test_scaled)
    y_prob_unsup = svm_unsup.predict_proba(X_unsup_test_scaled)
    
    acc_unsup = accuracy_score(y_test, y_pred_unsup)
    f1_unsup = f1_score(y_test, y_pred_unsup, average='macro')
    
    # ROC AUC (micro-average)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    try:
        auc_unsup = roc_auc_score(y_test_bin, y_prob_unsup, average='micro')
    except:
        auc_unsup = None
    
    print(f"  Accuracy: {acc_unsup:.4f}")
    print(f"  F1 (macro): {f1_unsup:.4f}")
    if auc_unsup:
        print(f"  ROC AUC (micro): {auc_unsup:.4f}")
    
    results['unsupervised'] = {
        'accuracy': acc_unsup,
        'f1_macro': f1_unsup,
        'roc_auc_micro': auc_unsup
    }
    
    # Train SVM on semi-supervised embeddings
    print("\n" + "=" * 40)
    print("Training SVM on SEMI-SUPERVISED embeddings...")
    print("=" * 40)
    
    svm_semisup = SVC(kernel=args.kernel, C=args.C, probability=True, random_state=args.seed)
    svm_semisup.fit(X_semisup_train_scaled, y_train)
    
    y_pred_semisup = svm_semisup.predict(X_semisup_test_scaled)
    y_prob_semisup = svm_semisup.predict_proba(X_semisup_test_scaled)
    
    acc_semisup = accuracy_score(y_test, y_pred_semisup)
    f1_semisup = f1_score(y_test, y_pred_semisup, average='macro')
    
    try:
        auc_semisup = roc_auc_score(y_test_bin, y_prob_semisup, average='micro')
    except:
        auc_semisup = None
    
    print(f"  Accuracy: {acc_semisup:.4f}")
    print(f"  F1 (macro): {f1_semisup:.4f}")
    if auc_semisup:
        print(f"  ROC AUC (micro): {auc_semisup:.4f}")
    
    results['semisupervised'] = {
        'accuracy': acc_semisup,
        'f1_macro': f1_semisup,
        'roc_auc_micro': auc_semisup
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Unsupervised':<15} {'Semi-supervised':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {acc_unsup:<15.4f} {acc_semisup:<15.4f}")
    print(f"{'F1 (macro)':<20} {f1_unsup:<15.4f} {f1_semisup:<15.4f}")
    if auc_unsup and auc_semisup:
        print(f"{'ROC AUC (micro)':<20} {auc_unsup:<15.4f} {auc_semisup:<15.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results['config'] = {
        'kernel': args.kernel,
        'C': args.C,
        'test_size': args.test_size,
        'n_classes': n_classes,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
