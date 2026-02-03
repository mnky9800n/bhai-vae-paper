import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from catboost import CatBoostRegressor
import sys
import time
sys.path.insert(0, '.')
from models.vae import VAE, SemiSupervisedVAE, DistributionAwareScaler

np.random.seed(42)

FEATURE_COLS = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)', 
                'NGR total counts (cps)', 'R', 'G', 'B']
DATA_DIR = Path('data')
MODEL_DIR = Path('models')
LILY_DIR = Path('/home/mnky9800n/clawd/data/lily-datasets')

N_BOOTSTRAP = 100

print("="*60, flush=True)
print("FULL-SAMPLE BOOTSTRAP (no train/test split)", flush=True)
print("="*60, flush=True)

print("Loading training data and models...", flush=True)
train_df = pd.read_csv(DATA_DIR / 'vae_training_data_v2_20cm.csv')
X_raw = train_df[FEATURE_COLS].values
valid_mask = ~np.isnan(X_raw).any(axis=1)
train_df_valid = train_df[valid_mask].reset_index(drop=True)

scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X_raw[valid_mask])

model_unsup = VAE(input_dim=6, latent_dim=10)
model_unsup.load_state_dict(torch.load(MODEL_DIR / 'unsup.pt', map_location='cpu'))
model_unsup.eval()
model_semisup = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=139)
model_semisup.load_state_dict(torch.load(MODEL_DIR / 'semisup.pt', map_location='cpu'))
model_semisup.eval()

with torch.no_grad():
    X_t = torch.FloatTensor(X_scaled)
    emb_unsup = model_unsup.get_embeddings(X_t).numpy()
    emb_semisup = model_semisup.get_embeddings(X_t).numpy()

train_df_valid['idx'] = np.arange(len(train_df_valid))
print(f"Training samples: {len(train_df_valid):,}", flush=True)

def create_borehole_key(df):
    return df['Exp'].astype(str) + "-" + df['Site'].astype(str) + "-" + df['Hole'].astype(str)

datasets = {
    'AVS': 'Depth CSF-A (m)', 'CARB': 'Top depth CSF-A (m)', 'GE': 'Top depth CSF-A (m)',
    'ICP': 'Top depth CSF-A (m)', 'IW': 'Top depth CSF-A (m)', 'JR6A': 'Top depth CSF-A (m)',
    'KAPPA': 'Top depth CSF-A (m)', 'MAD': 'Depth CSF-A (m)', 'PEN': 'Depth CSF-A (m)',
    'PWB': 'Depth CSF-A (m)', 'PWC': 'Depth CSF-A (m)', 'SRMD': 'Depth CSF-A (m)',
    'TCON': 'Depth CSF-A (m)', 'TOR': 'Depth CSF-A (m)',
}

exclude_cols = {'Borehole_ID', 'Depth_Bin', 'idx', 'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 
                'A/W', 'Timestamp (UTC)', 'Instrument', 'Text ID', 'Test No.', 'Comments', 
                'Prefix', 'Principal', 'Suffix', 'Full Lithology', 'Simplified Lithology', 
                'Lithology Type', 'Latitude (DD)', 'Longitude (DD)', 'Water Depth (mbsl)',
                'Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B'}

def bootstrap_r2_full(y, X_unsup, X_semisup, n_boot=N_BOOTSTRAP):
    """Bootstrap with full-sample fit - measures embedding capacity"""
    r2_u_list, r2_s_list = [], []
    n = len(y)
    for b in range(n_boot):
        idx = np.random.choice(n, size=min(n, 50000), replace=True)
        y_b, Xu_b, Xs_b = y[idx], X_unsup[idx], X_semisup[idx]
        try:
            m_u = CatBoostRegressor(verbose=False, random_state=b, iterations=100)
            m_u.fit(Xu_b, y_b)
            r2_u_list.append(m_u.score(Xu_b, y_b))
            m_s = CatBoostRegressor(verbose=False, random_state=b, iterations=100)
            m_s.fit(Xs_b, y_b)
            r2_s_list.append(m_s.score(Xs_b, y_b))
        except:
            pass
    return r2_u_list, r2_s_list

results = []

# Depth baseline
print("\n[1/?] Depth baseline...", flush=True)
y = train_df_valid['Depth_Bin'].values
r2_u, r2_s = bootstrap_r2_full(y, emb_unsup, emb_semisup)
results.append({'variable': 'Depth (m)', 'n_samples': len(y),
    'r2_v267': np.mean(r2_u), 'r2_v267_lo': np.percentile(r2_u, 2.5), 'r2_v267_hi': np.percentile(r2_u, 97.5),
    'r2_v214': np.mean(r2_s), 'r2_v214_lo': np.percentile(r2_s, 2.5), 'r2_v214_hi': np.percentile(r2_s, 97.5)})
print(f"  Depth: unsup={np.mean(r2_u):.3f} [{np.percentile(r2_u,2.5):.3f}-{np.percentile(r2_u,97.5):.3f}], semi={np.mean(r2_s):.3f} [{np.percentile(r2_s,2.5):.3f}-{np.percentile(r2_s,97.5):.3f}]", flush=True)

ds_num = 2
for dataset_name, depth_col in datasets.items():
    filepath = LILY_DIR / f"{dataset_name}_DataLITH.csv"
    if not filepath.exists():
        continue
    print(f"\n[{ds_num}/?] {dataset_name}...", flush=True)
    ds_num += 1
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        continue
    df['Borehole_ID'] = create_borehole_key(df)
    if depth_col not in df.columns:
        continue
    
    merged = df.merge(train_df_valid[['Borehole_ID', 'Depth_Bin', 'idx']], on='Borehole_ID')
    merged['depth_diff'] = abs(merged[depth_col] - merged['Depth_Bin'])
    merged = merged[merged['depth_diff'] < 0.5]
    merged = merged.loc[merged.groupby(merged.index)['depth_diff'].idxmin()]
    
    if len(merged) < 100:
        continue
    print(f"  Matched {len(merged):,} samples", flush=True)
    
    numeric_cols = [c for c in df.columns if c not in exclude_cols and c != depth_col]
    var_count = 0
    for col in numeric_cols:
        if col not in merged.columns:
            continue
        try:
            vals = pd.to_numeric(merged[col], errors='coerce')
            mask = vals.notna() & np.isfinite(vals)
            if mask.sum() < 100:
                continue
            y = vals[mask].values
            idxs = merged.loc[mask, 'idx'].values.astype(int)
            Xu = emb_unsup[idxs]
            Xs = emb_semisup[idxs]
            r2_u, r2_s = bootstrap_r2_full(y, Xu, Xs)
            if len(r2_u) > 10:
                var_name = f"{dataset_name}: {col}"
                results.append({'variable': var_name, 'n_samples': len(y),
                    'r2_v267': np.mean(r2_u), 'r2_v267_lo': np.percentile(r2_u, 2.5), 'r2_v267_hi': np.percentile(r2_u, 97.5),
                    'r2_v214': np.mean(r2_s), 'r2_v214_lo': np.percentile(r2_s, 2.5), 'r2_v214_hi': np.percentile(r2_s, 97.5)})
                var_count += 1
        except:
            pass
    print(f"  ✓ {dataset_name}: {var_count} variables", flush=True)
    
    # Save partial results
    pd.DataFrame(results).to_csv('data/zeroshot_bootstrap_full.csv', index=False)

print(f"\n{'='*60}", flush=True)
print(f"Done! {len(results)} variables → data/zeroshot_bootstrap_full.csv", flush=True)
print(f"{'='*60}", flush=True)
