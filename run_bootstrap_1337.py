import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
OUTPUT_DIR = Path('figures')
LILY_DIR = Path('/home/mnky9800n/clawd/data/lily-datasets')

N_BOOTSTRAP = 100  # 1337 mode: 100 bootstrap iterations
MAX_SAMPLES = 50000

print(f"{'='*60}", flush=True)
print(f"1337 MODE: {N_BOOTSTRAP} bootstrap iterations per variable", flush=True)
print(f"{'='*60}", flush=True)

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

results = []
start_time = time.time()

def bootstrap_r2(y, X_unsup, X_semisup, n_boot=N_BOOTSTRAP):
    r2_u_list, r2_s_list = [], []
    n = len(y)
    for b in range(n_boot):
        idx = np.random.choice(n, size=min(n, MAX_SAMPLES), replace=True)
        y_b, Xu_b, Xs_b = y[idx], X_unsup[idx], X_semisup[idx]
        n_train = int(0.8 * len(y_b))
        perm = np.random.permutation(len(y_b))
        tr, te = perm[:n_train], perm[n_train:]
        try:
            m_u = CatBoostRegressor(verbose=False, random_state=b, iterations=100)
            m_u.fit(Xu_b[tr], y_b[tr])
            r2_u_list.append(m_u.score(Xu_b[te], y_b[te]))
            m_s = CatBoostRegressor(verbose=False, random_state=b, iterations=100)
            m_s.fit(Xs_b[tr], y_b[tr])
            r2_s_list.append(m_s.score(Xs_b[te], y_b[te]))
        except:
            pass
    return r2_u_list, r2_s_list

# Depth baseline
print("\n[1/?] Depth baseline...", flush=True)
y = train_df_valid['Depth_Bin'].values
r2_u, r2_s = bootstrap_r2(y, emb_unsup, emb_semisup)
results.append({'variable': 'Depth (m)', 'n_samples': len(y),
    'r2_v267': np.mean(r2_u), 'r2_v267_lo': np.percentile(r2_u, 2.5), 'r2_v267_hi': np.percentile(r2_u, 97.5),
    'r2_v214': np.mean(r2_s), 'r2_v214_lo': np.percentile(r2_s, 2.5), 'r2_v214_hi': np.percentile(r2_s, 97.5)})
print(f"    R²: unsup={np.mean(r2_u):.3f} [{np.percentile(r2_u,2.5):.3f}-{np.percentile(r2_u,97.5):.3f}], semi={np.mean(r2_s):.3f} [{np.percentile(r2_s,2.5):.3f}-{np.percentile(r2_s,97.5):.3f}]", flush=True)

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
        for alt in ['Depth CSF-A (m)', 'Top depth CSF-A (m)']:
            if alt in df.columns: depth_col = alt; break
        else: continue
    df['Depth_Bin'] = (df[depth_col] * 5).round() / 5
    merged = df.merge(train_df_valid[['Borehole_ID', 'Depth_Bin', 'idx']], on=['Borehole_ID', 'Depth_Bin'], how='inner')
    if len(merged) < 100: continue
    print(f"    Matched {len(merged):,} samples", flush=True)
    var_count = 0
    for col in merged.columns:
        if col in exclude_cols or 'offset' in col.lower() or 'depth' in col.lower(): continue
        try: values = pd.to_numeric(merged[col], errors='coerce')
        except: continue
        valid = values.notna()
        if valid.sum() < 100: continue
        y = values[valid].values.astype(float)
        if np.std(y) < 1e-6: continue
        indices = merged.loc[valid, 'idx'].values.astype(int)
        r2_u, r2_s = bootstrap_r2(y, emb_unsup[indices], emb_semisup[indices])
        if len(r2_u) < 10: continue
        results.append({'variable': f"{dataset_name}: {col}", 'n_samples': len(y),
            'r2_v267': np.mean(r2_u), 'r2_v267_lo': np.percentile(r2_u, 2.5), 'r2_v267_hi': np.percentile(r2_u, 97.5),
            'r2_v214': np.mean(r2_s), 'r2_v214_lo': np.percentile(r2_s, 2.5), 'r2_v214_hi': np.percentile(r2_s, 97.5)})
        var_count += 1
    print(f"    Processed {var_count} variables", flush=True)
    pd.DataFrame(results).to_csv(DATA_DIR / 'zeroshot_bootstrap_partial.csv', index=False)

elapsed = time.time() - start_time
print(f"\n{'='*60}", flush=True)
print(f"DONE! {len(results)} variables in {elapsed/60:.1f} min", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(DATA_DIR / 'zeroshot_bootstrap_1337.csv', index=False)

# Plot
df = df_r[(df_r['r2_v267'] > -1) & (df_r['r2_v214'] > -1) & (df_r['r2_v267'] < 1.1) & (df_r['r2_v214'] < 1.1)]
fig, ax = plt.subplots(figsize=(10, 10))
x, y = df['r2_v214'].values, df['r2_v267'].values
x_lo, x_hi = df['r2_v214_lo'].values, df['r2_v214_hi'].values
y_lo, y_hi = df['r2_v267_lo'].values, df['r2_v267_hi'].values
n_samples = df['n_samples'].values
sizes = 20 + 80 * (np.log10(np.clip(n_samples, 100, 1e6)) - 2) / 4
c_semi, c_unsup = '#20B2AA', '#F08080'
colors = [c_semi if xi > yi else c_unsup for xi, yi in zip(x, y)]
for i in range(len(x)):
    ax.errorbar(x[i], y[i], xerr=[[max(0,x[i]-x_lo[i])],[max(0,x_hi[i]-x[i])]], 
                yerr=[[max(0,y[i]-y_lo[i])],[max(0,y_hi[i]-y[i])]], fmt='o', 
                markersize=np.sqrt(sizes[i]/3), color=colors[i], alpha=0.6, elinewidth=0.5, capsize=0)
ax.plot([-0.2, 1.05], [-0.2, 1.05], 'k--', lw=1.5, alpha=0.5)
ax.set_xlabel('Semi-supervised R²', fontsize=14); ax.set_ylabel('Unsupervised R²', fontsize=14)
ax.set_xlim(-0.2, 1.05); ax.set_ylim(-0.2, 1.05); ax.set_aspect('equal')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c_semi,markersize=10,label='Semi-sup higher'),
                   Line2D([0],[0],marker='o',color='w',markerfacecolor=c_unsup,markersize=10,label='Unsup higher')], loc='lower right')
sax = fig.add_axes([0.15, 0.78, 0.1, 0.12]); sax.axis('off')
for ss, lab, yp in [(100,'100',0.7),(10000,'10k',0.2)]:
    sax.scatter(0.3, yp, s=20+80*(np.log10(ss)-2)/4, c='gray', alpha=0.7)
    sax.text(0.6, yp, lab, va='center', fontsize=9)
sax.text(0.45, 0.95, 'n', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_r2_unsup_vs_semi.png', dpi=300, bbox_inches='tight')
print(f"Semi wins: {sum(1 for xi,yi in zip(x,y) if xi>yi)}/{len(x)}", flush=True)
print("SAVED fig_r2_unsup_vs_semi.png 💀🔥", flush=True)
