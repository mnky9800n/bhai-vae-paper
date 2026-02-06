# Scripts

**Last updated:** 2026-02-05

## scripts/ Directory

### train_vae.py

**Purpose:** Train a single VAE model (unsupervised or semi-supervised).

**Inputs:**
- `--model`: `unsupervised` or `semisupervised`
- `--data`: Path to training CSV (default: `data/vae_training_data_v2_20cm.csv`)
- `--output`: Path for saved model `.pt` file
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 1e-3)
- `--beta`: KL weight (default: 1.0)
- `--alpha`: Classification weight, semi-supervised only (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--device`: `cpu` or `cuda`

**Outputs:**
- Model checkpoint (`.pt`)
- Training history (`.json`)

**Example:**
```bash
python scripts/train_vae.py --model unsupervised --output models/unsup.pt
python scripts/train_vae.py --model semisupervised --output models/semisup.pt --alpha 0.1
```

---

### generate_all_figures.py

**Purpose:** Train hybrid-loss models and generate all paper figures. This is the **main figure generation script**.

**Inputs:**
- `--skip-training`: Skip training, load existing `model_*_hybrid.pt`
- Hardcoded: reads `data/vae_training_data_v2_20cm.csv`

**Outputs:**
- `models/model_unsup_hybrid.pt`
- `models/model_semisup_hybrid.pt`
- `models/embeddings_hybrid.npz`
- `figures/fig_reconstruction_scatter.png`
- `figures/fig_roc_comparison.png`
- `figures/fig_umap_lithology.png`
- `figures/fig_zeroshot_scatter.png`
- `figures/fig_generated_variables_grid.png`

**Example:**
```bash
uv run python scripts/generate_all_figures.py
uv run python scripts/generate_all_figures.py --skip-training
```

**Note:** Uses hybrid training with masking, KL annealing. Different hyperparameters than `train_vae.py`.

---

### generate_figures.py

**Purpose:** Generate paper figures using pre-trained models (older version).

**Inputs:**
- Hardcoded: reads `models/unsup.pt`, `models/semisup.pt`
- Hardcoded: reads `data/vae_training_data_v2_20cm.csv`
- Hardcoded: reads `data/zeroshot_results.csv` (if exists)

**Outputs:**
- `figures/fig_zeroshot_scatter.png`
- `figures/fig_reconstruction_scatter.png`
- `figures/fig_roc_comparison.png`

**Note:** Uses original models, not hybrid. Superseded by `generate_all_figures.py`.

---

### generate_embeddings.py

**Purpose:** Generate embedding table from trained models, attach to original data.

**Inputs:**
- `--unsup-model`: Path to unsupervised model
- `--semisup-model`: Path to semi-supervised model
- `--data`: Path to training CSV
- `--output`: Path for output CSV
- `--latent-dim`: Latent dimension (default: 10)

**Outputs:**
- CSV with original columns + 20 embedding columns (`unsup_emb_0..9`, `semisup_emb_0..9`)

**Example:**
```bash
python scripts/generate_embeddings.py \
    --unsup-model models/unsup.pt \
    --semisup-model models/semisup.pt \
    --data data/vae_training_data_v2_20cm.csv \
    --output data/embeddings.csv
```

---

### run_svm.py

**Purpose:** Run SVM classification on VAE embeddings to compare unsupervised vs semi-supervised.

**Inputs:**
- `--embeddings`: Path to embeddings CSV (from `generate_embeddings.py`)
- `--output`: Path for results JSON
- `--lithology-col`: Column name for labels (default: `Principal`)
- `--test-size`: Test fraction (default: 0.2)
- `--seed`: Random seed (default: 42)
- `--max-samples`: Max samples to use (default: 50000)
- `--kernel`: SVM kernel (default: `rbf`)
- `--C`: Regularization (default: 1.0)

**Outputs:**
- JSON with accuracy, F1, ROC AUC for both embedding types

**Example:**
```bash
python scripts/run_svm.py --embeddings data/embeddings.csv --output results/svm_results.json
```

---

### fig_r2_scatter_with_ci.py

**Purpose:** Generate R² scatter plot comparing unsupervised vs semi-supervised with 95% CI error bars.

**Inputs:**
- Hardcoded: reads `data/zeroshot_scatter_results_full.csv`
- Falls back to: `/home/mnky9800n/clawd/bhai-analysis/zeroshot_scatter_results_full.csv`

**Outputs:**
- `figures/fig_r2_unsup_vs_semi.png`
- `data/zeroshot_scatter_results_with_ci.csv`

**Example:**
```bash
uv run python scripts/fig_r2_scatter_with_ci.py
```

---

### remake_reconstruction_scatter.py

**Purpose:** Regenerate reconstruction scatter without subsampling.

**Inputs:**
- Hardcoded: reads `models/unsup.pt`, `models/semisup.pt`
- Hardcoded: reads `data/vae_training_data_v2_20cm.csv`

**Outputs:**
- `figures/fig_reconstruction_scatter.png` (overwrites)
- Prints per-variable R² scores

**Note:** Uses original models, not hybrid. May produce different results than `generate_all_figures.py`.

---

### run_bootstrap.sh

**Purpose:** Shell script to run multiple bootstrap training iterations.

**Inputs:**
- `$1`: Number of bootstrap iterations (default: 100)
- `$2`: Epochs per iteration (default: 100)

**Outputs:**
- `models/bootstrap/unsupervised/model_1.pt` ... `model_N.pt`
- `models/bootstrap/semisupervised/model_1.pt` ... `model_N.pt`

**Example:**
```bash
bash scripts/run_bootstrap.sh 100 100
```

---

## Root Directory Scripts

### run_bootstrap_1337.py

**Purpose:** Run 100-iteration bootstrap evaluation to compare unsupervised vs semi-supervised embeddings on zero-shot prediction of external variables.

**Inputs:**
- Hardcoded: `data/vae_training_data_v2_20cm.csv`
- Hardcoded: `models/unsup.pt`, `models/semisup.pt`
- **External:** `/home/mnky9800n/clawd/data/lily-datasets/*.csv`

**Outputs:**
- `data/zeroshot_bootstrap_1337.csv`
- `data/zeroshot_bootstrap_partial.csv` (intermediate)
- `figures/fig_r2_unsup_vs_semi.png`

**Method:**
- For each external variable, bootstrap 100 times with train/test split
- Train CatBoost regressor on embeddings to predict variable
- Compute R² with 95% CI

**Example:**
```bash
uv run python run_bootstrap_1337.py
```

---

### run_bootstrap_full.py

**Purpose:** Full-sample bootstrap (no train/test split) - measures embedding capacity rather than generalization.

**Inputs:**
- Same as `run_bootstrap_1337.py`

**Outputs:**
- `data/zeroshot_bootstrap_full.csv`

**Method:**
- Similar to 1337, but trains and evaluates on same data (no split)

---

### run_figures.sh

**Purpose:** Wrapper script to run figure generation via uv.

**Content:**
```bash
#!/bin/bash
uv run python scripts/generate_all_figures.py "$@"
```

---

## Notebooks

### notebooks/paper_figures.ipynb

**Purpose:** Interactive figure generation, including geographic maps.

**Requires:** `cartopy` (for expedition maps)

**Generates:**
- `fig_lily_dataset.png`
- `fig_lily_expedition_map.png`
- `fig_lily_lithology_counts.png`
- `fig_lily_variables_dist.png`

**Note:** These figures are NOT generated by any Python script. Must run notebook to regenerate.

---

## Script Dependency Graph

```
Data Flow:
                    
vae_training_data_v2_20cm.csv
           │
           ▼
    ┌──────────────┐
    │ train_vae.py │ ─────────► unsup.pt, semisup.pt
    └──────────────┘
           │
           ▼
    ┌────────────────────────┐
    │ generate_embeddings.py │ ─────► embeddings.csv
    └────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ run_svm.py  │ ─────────► svm_results.json
    └─────────────┘

Figure Generation:

    ┌──────────────────────────┐
    │ generate_all_figures.py  │ ─────► model_*_hybrid.pt
    │ (trains + generates)     │ ─────► embeddings_hybrid.npz
    └──────────────────────────┘ ─────► fig_reconstruction_scatter.png
                                 ─────► fig_roc_comparison.png
                                 ─────► fig_umap_lithology.png
                                 ─────► fig_zeroshot_scatter.png
                                 ─────► fig_generated_variables_grid.png

    ┌───────────────────────┐
    │ run_bootstrap_1337.py │ ─────► zeroshot_bootstrap_1337.csv
    │ (uses original models)│ ─────► fig_r2_unsup_vs_semi.png
    └───────────────────────┘

    ┌──────────────────────────┐
    │ paper_figures.ipynb      │ ─────► fig_lily_*.png (4 figures)
    └──────────────────────────┘
```
