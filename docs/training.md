# Training

**Last updated:** 2026-02-05

## Training Methods

This repository contains **two different training approaches** that produce different models:

| Method | Script | Models Produced | Key Differences |
|--------|--------|-----------------|-----------------|
| Standard | `scripts/train_vae.py` | `unsup.pt`, `semisup.pt` | Simple VAE loss, fixed β |
| Hybrid | `scripts/generate_all_figures.py` | `model_*_hybrid.pt` | Masking, KL annealing, gradient clipping |

---

## Standard Training

### Script: `scripts/train_vae.py`

**Hyperparameters:**

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Epochs | 100 | `--epochs` |
| Batch size | 256 | `--batch-size` |
| Learning rate | 1e-3 | `--lr` |
| β (KL weight) | 1.0 | `--beta` |
| α (class weight) | 0.1 | `--alpha` |
| Random seed | 42 | `--seed` |
| Optimizer | Adam | (hardcoded) |

**Loss function:**

```python
# Unsupervised
loss = MSE(recon, x, reduction='sum') + β * KL_divergence

# Semi-supervised
loss = MSE(recon, x, reduction='sum') + β * KL + α * CrossEntropy(logits, labels)
```

**Training loop:**
```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = model.loss_function(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
```

**Commands to reproduce original models:**

```bash
# Unsupervised
python scripts/train_vae.py \
    --model unsupervised \
    --data data/vae_training_data_v2_20cm.csv \
    --output models/unsup.pt \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --beta 1.0 \
    --seed 42

# Semi-supervised
python scripts/train_vae.py \
    --model semisupervised \
    --data data/vae_training_data_v2_20cm.csv \
    --output models/semisup.pt \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --beta 1.0 \
    --alpha 0.1 \
    --seed 42
```

---

## Hybrid Training

### Script: `scripts/generate_all_figures.py`

**Hyperparameters (hardcoded in script):**

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Latent dim | 10 |
| Hidden dims | [64, 32] |
| Mask ratio | 0.1 (10%) |
| Mask weight | 0.5 |
| β start | 1e-6 |
| β end | 0.5 |
| β anneal epochs | 30 |
| α (class weight) | 0.5 |
| Gradient clip | 1.0 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |

**Differences from standard:**

1. **Input masking:** 10% of input features randomly zeroed each batch
2. **KL annealing:** β ramps linearly from 1e-6 to 0.5 over 30 epochs
3. **Masked loss:** Extra term penalizing reconstruction of masked positions
4. **Higher α:** 0.5 vs 0.1 for classification weight
5. **Gradient clipping:** Max norm 1.0
6. **LR scheduler:** Reduces LR when loss plateaus

**Loss function:**

```python
# Create mask
mask = torch.rand_like(x) < 0.1  # 10% masked
x_masked = x.clone()
x_masked[mask] = 0

# Forward pass with masked input
recon, mu, logvar = model(x_masked)

# Losses (note: mean reduction, not sum)
recon_loss = MSE(recon, x, reduction='mean')
masked_loss = MSE(recon[mask], x[mask], reduction='mean')
kl_loss = -0.5 * mean(1 + logvar - mu² - exp(logvar))

# β anneals over first 30 epochs
beta = beta_start + (beta_end - beta_start) * min(epoch / 30, 1.0)

# Total loss
loss = recon_loss + 0.5 * masked_loss + beta * kl_loss

# Semi-supervised adds classification
loss = loss + 0.5 * CrossEntropy(logits, labels)
```

**Commands to reproduce hybrid models:**

```bash
# Train both models and generate figures
uv run python scripts/generate_all_figures.py

# Or just regenerate figures with existing models
uv run python scripts/generate_all_figures.py --skip-training
```

---

## Reproducibility

### Random Seeds

| Location | Seed | Scope |
|----------|------|-------|
| `train_vae.py` | 42 (default) | `torch.manual_seed`, `np.random.seed` |
| `generate_all_figures.py` | 42 | All RNGs + CUDA |

### CUDA Determinism

`generate_all_figures.py` sets:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`train_vae.py` does **not** set these.

### To Ensure Full Reproducibility

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Training Data Statistics

| Property | Value |
|----------|-------|
| Total samples | 238,506 |
| Features | 6 |
| Lithology classes | 139 |
| Batch size | 256 |
| Batches per epoch | ~931 |

---

## Hardware

Training is fast (minutes on GPU, tens of minutes on CPU).

| Device | Training time (100 epochs) |
|--------|---------------------------|
| CUDA GPU | ~2-5 minutes |
| CPU | ~20-30 minutes |

Script auto-detects:
```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

---

## Model Checkpoints

Models are saved as PyTorch state dicts:

```python
# Save
torch.save(model.state_dict(), 'models/unsup.pt')

# Load
model = VAE(input_dim=6, latent_dim=10, hidden_dims=[64, 32])
model.load_state_dict(torch.load('models/unsup.pt', map_location='cpu'))
```

---

## Training History

`train_vae.py` saves training history:

```python
# Saved to {output_path}.json
history = {
    'loss': [...],      # Per-epoch total loss
    'recon': [...],     # Per-epoch reconstruction loss  
    'kl': [...],        # Per-epoch KL divergence
    'class': [...]      # Per-epoch classification loss (semi-supervised only)
}
```

`generate_all_figures.py` does **not** save history.

---

## Summary of Key Differences

| Aspect | Standard | Hybrid |
|--------|----------|--------|
| β | Fixed 1.0 | Anneals 1e-6 → 0.5 |
| α | 0.1 | 0.5 |
| Masking | None | 10% input masking |
| Loss reduction | sum | mean |
| Gradient clipping | None | 1.0 |
| LR scheduler | None | ReduceLROnPlateau |
| Determinism | Seed only | Seed + CUDA determinism |
| History saved | Yes | No |
