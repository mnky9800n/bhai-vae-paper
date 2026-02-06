# Unified Paper Notebook Specification

## Overview
Create a comprehensive Jupyter notebook (`notebooks/bhai_vae_paper.ipynb`) that:
1. Downloads and processes data
2. Trains both models with reproducible seeds
3. Generates all paper figures
4. Uses 10% masked encoding during training

## Requirements

### 1. Data Pipeline

#### 1.1 Download from Zenodo (if not exists)
- GRA_DataLITH.csv
- MS_DataLITH.csv  
- NGR_DataLITH.csv
- RSC_DataLITH.csv (for L*a*b* or RGB)
- Download to `data/raw/` if not present
- Zenodo DOI: 10.5281/zenodo.7484524

#### 1.2 Data Processing
- Bin measurements to 20cm depth intervals
- Merge datasets (inner join for complete samples)
- Save to `data/processed/vae_training_data.csv`

### 2. Descriptive Figures (Before Training)

#### 2.1 fig_lily_expedition_map.png
- World map with expedition locations
- Color by expedition or measurement type

#### 2.2 fig_lily_dataset.png
- Dataset overview visualization
- Sample counts, borehole counts, depth coverage

#### 2.3 fig_lily_lithology_counts.png
- Bar chart of lithology class frequencies
- Top 30 or all classes

#### 2.4 fig_lily_variables_dist.png
- 6-panel distribution plot
- One panel per input feature (GRA, MS, NGR, R, G, B)

### 3. Model Training

#### 3.1 Reproducibility
- Set random seeds for: numpy, torch, python random
- SEED = 42 (or configurable)

#### 3.2 Data Scaling
- DistributionAwareScaler (signed log for MS/NGR, log1p for RGB)
- StandardScaler normalization

#### 3.3 Masked Encoding (10%)
- During training, randomly mask 10% of input features
- Replace masked values with 0 (after scaling)
- Teaches model to handle missing data

#### 3.4 Model A: Unsupervised VAE
Architecture:
- Input: 6D
- Encoder: [64, 32] with BatchNorm
- Latent: 10D
- Decoder: [32, 64] with BatchNorm
- Output: 6D

Training:
- β-annealing: 1e-10 → 0.075 over 50 epochs
- 100 epochs total
- Adam optimizer, lr=1e-3
- Batch size 256

Save: `models/model_unsup.pt`

#### 3.5 Model B: Semi-Supervised VAE
Architecture:
- Same encoder/decoder as Model A
- Classification head: latent(10) → 64 → Dropout(0.3) → 139 classes

Training:
- Same β-annealing as Model A
- α-annealing: 0 → 0.01 over 50 epochs
- 100 epochs total

Save: `models/model_semisup.pt`

#### 3.6 Interactive Training Visualization
- Use tqdm progress bars
- Live loss plot (optional: use livelossplot or matplotlib animation)
- Show: total loss, reconstruction loss, KL loss, (classification loss for semisup)

### 4. Model Evaluation

#### 4.1 Unsupervised Model Evaluation
- Generate embeddings for test set
- Train SVM classifier on embeddings
- Report: accuracy, precision, recall, F1
- Compute ARI for k-means clustering (k=10, 12, 15, 20)

#### 4.2 Semi-Supervised Model Evaluation  
- Generate embeddings for test set
- Get classification predictions from head
- Report: accuracy, AUC (macro and per-class), precision, recall, F1
- Compute ARI for k-means clustering

### 5. Evaluation Figures (After Training)

#### 5.1 fig_reconstruction_scatter.png
- 6-panel scatter: true vs reconstructed
- One panel per feature
- Include R² values

#### 5.2 fig_roc_comparison.png
- ROC curves comparing unsupervised (SVM) vs semi-supervised (head)
- Macro-average curves
- AUC values in legend

#### 5.3 fig_r2_unsup_vs_semi.png
- Scatter plot: R² unsupervised vs R² semi-supervised
- Per-feature comparison
- Diagonal reference line

#### 5.4 fig_umap_lithology.png
- UMAP of latent space
- Color by lithology
- Two panels: unsupervised / semi-supervised

### 6. Network Diagrams

#### 6.1 Unsupervised VAE diagram
- Box diagram showing architecture
- Include layer sizes and activations
- Generate from PyTorch model spec

#### 6.2 Semi-Supervised VAE diagram
- Same as above + classification head
- Show the branching from latent space

### 7. Code Organization

```python
# Section headers in notebook:
# 1. Setup and Imports
# 2. Configuration
# 3. Data Download
# 4. Data Processing
# 5. Descriptive Figures
# 6. Model Definitions
# 7. Training Utilities (masked encoding, annealing)
# 8. Train Unsupervised VAE
# 9. Train Semi-Supervised VAE
# 10. Evaluation Utilities
# 11. Evaluate Models
# 12. Evaluation Figures
# 13. Network Diagrams
# 14. Summary and Export
```

### 8. Dependencies
- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- umap-learn
- tqdm
- requests (for Zenodo download)
- cartopy (for map)

### 9. Commit Strategy
After each section is complete and tested:
1. Run the notebook to verify
2. Clear outputs
3. Commit with descriptive message

Example commits:
- "Add data download and processing pipeline"
- "Add descriptive figures (expedition map, dataset overview)"
- "Add model definitions with masked encoding"
- "Add unsupervised VAE training with β-annealing"
- "Add semi-supervised VAE training with α-annealing"
- "Add evaluation pipeline (SVM, metrics)"
- "Add evaluation figures (reconstruction, ROC, UMAP)"
- "Add network diagrams"
- "Final cleanup and documentation"
