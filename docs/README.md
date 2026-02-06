# BHAI VAE Paper - Documentation

**Last audited:** 2026-02-05

## Overview

This repository contains VAE and Semi-Supervised VAE models for generating embeddings from IODP core physical property data. The embeddings are used for downstream tasks like lithology classification and zero-shot prediction of other geochemical variables.

## Quick Links

| Document | Contents |
|----------|----------|
| [models.md](models.md) | Model architectures, saved weights, parameter counts |
| [scripts.md](scripts.md) | Each script with purpose, inputs, outputs |
| [figures.md](figures.md) | Figure inventory with generation provenance |
| [data.md](data.md) | Data files and preprocessing pipeline |
| [training.md](training.md) | Training procedures and hyperparameters |
| [issues.md](issues.md) | Inconsistencies and unknowns |

## Directory Structure

```
bhai-vae-paper/
├── docs/                   # This documentation
├── data/                   # Training data and evaluation results
│   ├── vae_training_data_v2_20cm.csv   # Main training set
│   ├── embeddings.csv      # Generated embeddings
│   └── zeroshot_*.csv      # Bootstrap evaluation results
├── figures/                # Generated paper figures
├── models/                 # Model definitions and checkpoints
│   ├── vae.py              # Architecture definitions (source of truth)
│   ├── unsup.pt            # Unsupervised VAE weights
│   ├── semisup.pt          # Semi-supervised VAE weights
│   └── model_*_hybrid.pt   # Hybrid-loss trained models
├── scripts/                # Python and shell scripts
├── notebooks/              # Jupyter notebooks
│   └── paper_figures.ipynb # Interactive figure generation
├── results/                # SVM classification results
├── run_bootstrap_*.py      # Bootstrap evaluation scripts
└── run_figures.sh          # Figure generation wrapper
```

## Related Repositories

### bhai-analysis (sibling directory)

**Location:** `/home/mnky9800n/clawd/bhai-analysis`

Development/experimentation repo that predates bhai-vae-paper. Contains:
- Earlier versions of figure generation scripts
- Bootstrap training experiments
- Duplicate copies of training data (identical MD5)
- Some figures that were copied to bhai-vae-paper

**Key files that originated here:**
- `zeroshot_scatter_results_full.csv` (referenced by `fig_r2_scatter_with_ci.py`)
- Earlier model weights (`model_unsup_hybrid.pt`, `model_semisup_hybrid.pt`)
- Various experimental training scripts

**Relationship:** bhai-vae-paper is the "clean" paper-ready version; bhai-analysis is the experimental sandbox.

### External Data Dependencies

**LILY Datasets:** `/home/mnky9800n/clawd/data/lily-datasets/`

Contains IODP LIMS data exports:
- `AVS_DataLITH.csv`, `CARB_DataLITH.csv`, `GE_DataLITH.csv`, etc.
- Used by `run_bootstrap_1337.py` and `run_bootstrap_full.py` for zero-shot evaluation
- ~11GB total

## Git Remote

```
origin  https://github.com/mnky9800n/bhai-vae-paper.git
```

## Quick Start

```bash
# Install dependencies
uv sync

# Generate all figures (trains hybrid models first)
uv run python scripts/generate_all_figures.py

# Or skip training and use existing models
uv run python scripts/generate_all_figures.py --skip-training

# Run bootstrap evaluation (requires LILY data)
uv run python run_bootstrap_1337.py
```

## Key Concepts

### Two Training Approaches

1. **Standard Training** (`scripts/train_vae.py`): Simple VAE loss with fixed β=1.0
2. **Hybrid Training** (`scripts/generate_all_figures.py`): Adds input masking, KL annealing, gradient clipping

### Two Model Sets

1. **Original models** (`unsup.pt`, `semisup.pt`): Trained Feb 1 with standard training
2. **Hybrid models** (`model_*_hybrid.pt`): Trained Feb 5 with hybrid training

Most recent figures use hybrid models. See [issues.md](issues.md) for details on inconsistencies.
