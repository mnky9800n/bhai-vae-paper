# BHAI VAE Paper

Variational Autoencoder models for the BHAI (IODP Core Physical Properties) paper.

## Repository Structure

```
bhai-vae-paper/
├── models/
│   ├── vae.py              # VAE and SemiSupervisedVAE architectures
│   └── __init__.py
├── scripts/
│   ├── train_vae.py        # Train a single model
│   ├── run_bootstrap.sh    # Bootstrap training for both models
│   ├── generate_embeddings.py  # Generate embeddings table
│   └── run_svm.py          # SVM classification on embeddings
├── notebooks/
│   └── paper_figures.ipynb # Generate all paper figures
├── data/
│   └── vae_training_data_v2_20cm.csv  # Training data (not in repo)
└── figures/                # Output figures
```

## Setup

```bash
pip install torch numpy pandas scikit-learn catboost matplotlib
```

## Usage

### 1. Train Models

Train a single model:
```bash
python scripts/train_vae.py --model unsupervised --output models/unsup.pt
python scripts/train_vae.py --model semisupervised --output models/semisup.pt --alpha 0.1
```

Bootstrap training (100 iterations):
```bash
bash scripts/run_bootstrap.sh 100 100
```

### 2. Generate Embeddings

```bash
python scripts/generate_embeddings.py \
    --unsup-model models/unsup.pt \
    --semisup-model models/semisup.pt \
    --data data/vae_training_data_v2_20cm.csv \
    --output data/embeddings.csv
```

### 3. Run SVM Classification

```bash
python scripts/run_svm.py \
    --embeddings data/embeddings.csv \
    --output results/svm_results.json
```

### 4. Generate Figures

Open `notebooks/paper_figures.ipynb` in Jupyter and run all cells.

## Model Architectures

### Unsupervised VAE (v2.6.7)
- Encoder: input(6) → 64 → 32 → latent(10)
- Decoder: latent(10) → 32 → 64 → output(6)
- BatchNorm after each hidden layer

### Semi-Supervised VAE (v2.14)
- Same encoder/decoder as unsupervised
- Classification head: latent(10) → 64 → n_classes(139)
- Loss: Reconstruction + β×KL + α×Classification (α=0.1)

## Features

Input features (6D):
1. Bulk density (GRA)
2. Magnetic susceptibility (instr. units)
3. NGR total counts (cps)
4. R (red channel)
5. G (green channel)
6. B (blue channel)

Preprocessing:
- Signed log transform: Mag susc, NGR
- Log1p transform: R, G, B
- StandardScaler on all
