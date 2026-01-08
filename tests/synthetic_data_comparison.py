"""
Synthetic Data Generator Comparison: BN-BGAN vs CTGAN
======================================================

This module compares two state-of-the-art synthetic data generators:
1. BN-BGAN (Bayesian Network-Augmented BGAN) - our method
2. CTGAN (Conditional Tabular GAN) - literature baseline

Uses the same evaluation pipeline and dataset loading logic from main.py.
Evaluation metrics: MMD, Diversity (Silhouette Score), Indistinguishability (Classification Accuracy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io as arff
import os
import warnings
import sys

warnings.filterwarnings('ignore')

# Ensure repo root is on sys.path so local packages can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import SDG methods
try:
    from bn_bgan.bn_bgan_sdg import BN_AUG_SDG
    HAS_BN_BGAN = True
except Exception as e:
    print(f"Warning: Could not import BN_AUG_SDG: {e}")
    HAS_BN_BGAN = False

try:
    from ctgan import CTGAN
    HAS_CTGAN = True
except Exception as e:
    print(f"Warning: Could not import CTGAN: {e}")
    HAS_CTGAN = False

try:
    from bgan.synthesizers.tvae import TVAE
    HAS_TVAE = True
except Exception as e:
    print(f"Warning: Could not import TVAE: {e}")
    HAS_TVAE = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score

# ==========================
# Configuration
# ==========================

DATASETS = [
    {"name": "heart", "path": "new_datasets/baseline_heart_disease_dataset"},
    {"name": "cancer", "path": "datasets/Cancer_Dataset.arff"},
    {"name": "blood", "path": "datasets/Blood_Dataset.arff"},
]

N_REPEATS = 5
RANDOM_SEED = 42
EPOCHS = 100

OUTPUT_DIR = 'outputs/synthetic_data_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Data Loading (from main.py)
# ==========================

def load_arff_flex(fp):
    """Flexible ARFF loader from main.py"""
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read()

    parts = txt.split('\n@DATA')
    if len(parts) < 2:
        parts = txt.split('\n@data')
    if len(parts) < 2:
        raise ValueError('No @DATA section found')

    header = parts[0]
    data_section = parts[1]

    cols = []
    categorical_attributes = {}
    for line in header.splitlines():
        line = line.strip()
        if line.upper().startswith('@ATTRIBUTE'):
            parts_attr = line.split(None, 2)
            if len(parts_attr) >= 3:
                name = parts_attr[1]
                type_spec = parts_attr[2]
                
                if name.startswith('"') and name.endswith('"'):
                    name = name[1:-1]
                
                if type_spec.startswith('{') and type_spec.endswith('}'):
                    values = [v.strip().strip('"\'') for v in type_spec[1:-1].split(',')]
                    categorical_attributes[name] = values
                
                cols.append(name)

    data_lines = [l.strip() for l in data_section.splitlines() if l.strip() and not l.strip().startswith('%')]
    rows = []
    for l in data_lines:
        vals = [v.strip().strip('"') for v in l.split(',')]
        rows.append(vals)

    df = pd.DataFrame(rows, columns=cols)

    for col in df.columns:
        if col in categorical_attributes:
            if set(v.lower() for v in categorical_attributes[col]) == {'true', 'false'}:
                df[col] = df[col].apply(lambda x: str(x).lower() == 'true')
            else:
                df[col] = df[col].astype(str)
                values = categorical_attributes[col]
                value_map = {str(v).lower(): v for v in values} | {str(v).upper(): v for v in values} | {str(v): v for v in values}
                df[col] = df[col].apply(lambda x: value_map.get(str(x).strip(), np.nan))
        else:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

def load_dataset(dataset_path):
    """Load dataset from ARFF or pickle/CSV format"""
    if dataset_path.endswith('.arff'):
        data = load_arff_flex(dataset_path)
        for col in data.select_dtypes([object]):
            data[col] = data[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    else:
        if os.path.exists(dataset_path + '.csv'):
            try:
                data = pd.read_csv(dataset_path + '.csv')
            except Exception as e:
                # Try with different separators
                try:
                    data = pd.read_csv(dataset_path + '.csv', sep=';')
                except:
                    raise e
        elif os.path.exists(dataset_path + '.pkl'):
            data = pd.read_pickle(dataset_path + '.pkl')
        elif os.path.isfile(dataset_path):
            try:
                data = pd.read_pickle(dataset_path)
            except:
                try:
                    data = pd.read_csv(dataset_path)
                except:
                    try:
                        data = pd.read_csv(dataset_path, sep=';')
                    except:
                        raise FileNotFoundError(f"Could not load dataset: {dataset_path}")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    return data

# ==========================
# Evaluation Metrics
# ==========================

def compute_mmd(X, Y, kernel_bandwidth=None):
    """MMD using RBF kernel - lower is better"""
    if kernel_bandwidth is None:
        combined = np.vstack([X, Y])
        pairwise_dists = np.sqrt(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2))
        kernel_bandwidth = np.median(pairwise_dists[pairwise_dists > 0])
    
    K_XX = rbf_kernel(X, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    K_YY = rbf_kernel(Y, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    K_XY = rbf_kernel(X, Y, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return max(0, mmd)

def calculate_diversity(data):
    """Silhouette score - higher is better"""
    try:
        if len(data) < 2:
            return 0.0
        n_clusters = min(5, len(data) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        if len(np.unique(labels)) > 1:
            return silhouette_score(data, labels)
        return 0.0
    except:
        return 0.0

def calculate_indistinguishability(real, synthetic):
    """Classification accuracy - lower is better"""
    try:
        X = np.vstack([real, synthetic])
        y = np.hstack([np.ones(len(real)), np.zeros(len(synthetic))])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        return np.mean(cross_val_score(clf, X, y, cv=5))
    except:
        return 0.5

def preprocess_numeric(data):
    """Extract and standardize numeric data"""
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        scaler = StandardScaler()
        return scaler.fit_transform(numeric_data.fillna(numeric_data.mean()))
    return None

# ==========================
# Model Training
# ==========================

def train_bn_bgan(real_data, discrete_columns):
    """Train BN-BGAN"""
    if not HAS_BN_BGAN:
        return None
    try:
        print("    [BN-BGAN] Training...", end=" ", flush=True)
        model = BN_AUG_SDG(epochs=EPOCHS, batch_size=32, use_kl_loss=True, use_uncertainty_loss=True)
        model.fit(real_data, discrete_columns)
        print("✓")
        return model
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return None

def sample_bn_bgan(model, n_samples):
    """Sample from BN-BGAN"""
    try:
        return model.sample(n_samples)
    except:
        return None

def train_bn_bgan(real_data, discrete_columns):
    """Train BN-BGAN using logic from imputation_tests/main.py"""
    if not HAS_BN_BGAN:
        return None
    try:
        print("    [BN-BGAN] Training...", end=" ", flush=True)
        model = BN_AUG_SDG(
            epochs=EPOCHS,
            batch_norm=True,
            embedding_dim=256,
            bn_influence=0.1,
            use_kl_loss=True,
            use_uncertainty_loss=True,
            optimizer_type="adam"
        )
        model.fit(real_data, discrete_columns)
        print("✓")
        return model
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")
        return None

def sample_bn_bgan(model, n_samples):
    """Sample from BN-BGAN"""
    try:
        synthetic = model.sample(n_samples)
        if isinstance(synthetic, np.ndarray):
            synthetic = pd.DataFrame(synthetic)
        if len(synthetic) > 0:
            return synthetic
        else:
            return None
    except Exception as e:
        print(f"Warning: BN-BGAN sampling error: {str(e)[:50]}")
        return None

def train_ctgan(real_data, discrete_columns):
    """Train CTGAN with standard literature parameters"""
    if not HAS_CTGAN:
        return None
    try:
        print("    [CTGAN] Training...", end=" ", flush=True)
        model = CTGAN(epochs=10, batch_size=200, discriminator_steps=5)
        model.fit(real_data, discrete_columns)
        print("✓")
        return model
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return None

def sample_ctgan(model, n_samples):
    """Sample from CTGAN with error handling"""
    try:
        synthetic = model.sample(n_samples)
        # Handle case where sample returns wrong shape
        if isinstance(synthetic, np.ndarray):
            synthetic = pd.DataFrame(synthetic)
        if len(synthetic) > 0:
            return synthetic
        else:
            return None
    except Exception as e:
        print(f"Warning: CTGAN sampling error: {str(e)[:50]}")
        return None

def train_tvae(real_data, discrete_columns):
    """Train TVAE with same parameters as CTGAN"""
    if not HAS_TVAE:
        return None
    try:
        print("    [TVAE] Training...", end=" ", flush=True)
        model = TVAE(epochs=10)
        model.fit(real_data, discrete_columns)
        print("✓")
        return model
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return None

def sample_tvae(model, n_samples):
    """Sample from TVAE with error handling"""
    try:
        synthetic = model.sample(n_samples)
        # Handle case where sample returns wrong shape
        if isinstance(synthetic, np.ndarray):
            synthetic = pd.DataFrame(synthetic)
        if len(synthetic) > 0:
            return synthetic
        else:
            return None
    except Exception as e:
        print(f"Warning: TVAE sampling error: {str(e)[:50]}")
        return None

# ==========================
# Main Comparison
# ==========================

print("\n" + "="*80)
print("SYNTHETIC DATA GENERATOR COMPARISON: BN-BGAN vs CTGAN")
print("="*80)

all_results = []

for dataset in DATASETS:
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset['name'].upper()}")
    print(f"{'='*80}")
    
    try:
        print(f"Loading {dataset['name']}...", end=" ", flush=True)
        data = load_dataset(dataset['path'])
        data = data.dropna()
        print(f"✓ ({len(data)} samples)")
    except Exception as e:
        print(f"✗ (Error: {e})")
        continue
    
    # Get discrete columns
    discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Split
    print(f"Splitting (70/30)...", end=" ", flush=True)
    train_data, eval_data = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
    print(f"✓ (Train: {len(train_data)}, Eval: {len(eval_data)})")
    
    # Subsample if needed
    if len(train_data) > 500:
        train_data = train_data.sample(n=500, random_state=RANDOM_SEED)
        eval_data = eval_data.sample(n=min(200, len(eval_data)), random_state=RANDOM_SEED)
        print(f"  Subsampled: Train {len(train_data)}, Eval {len(eval_data)}")
    
    # Get numeric eval data
    eval_numeric = preprocess_numeric(eval_data)
    
    # Methods to test
    methods = []
    if HAS_BN_BGAN:
        methods.append(('BN-BGAN', train_bn_bgan, sample_bn_bgan))
    if HAS_CTGAN:
        methods.append(('CTGAN', train_ctgan, sample_ctgan))
    if HAS_TVAE:
        methods.append(('TVAE', train_tvae, sample_tvae))
    
    if not methods:
        print("ERROR: No methods available")
        continue
    
    # Run methods
    for method_name, train_fn, sample_fn in methods:
        print(f"\n  {method_name}")
        print(f"  {'-'*60}")
        
        mmd_list, div_list, inds_list = [], [], []
        
        for run in range(N_REPEATS):
            print(f"    Run {run+1}/{N_REPEATS}: ", end="", flush=True)
            try:
                model = train_fn(train_data, discrete_columns)
                if model is None:
                    raise Exception("Training failed")
                
                synthetic = sample_fn(model, len(eval_data))
                if synthetic is None or len(synthetic) == 0:
                    raise Exception("Sampling failed")
                
                synthetic_numeric = preprocess_numeric(synthetic)
                
                if eval_numeric is not None and synthetic_numeric is not None:
                    mmd = compute_mmd(eval_numeric, synthetic_numeric)
                    div = calculate_diversity(synthetic_numeric)
                    inds = calculate_indistinguishability(eval_numeric, synthetic_numeric)
                    
                    mmd_list.append(mmd)
                    div_list.append(div)
                    inds_list.append(inds)
                    print("✓", end=" ", flush=True)
                else:
                    print("✗", end=" ", flush=True)
            except:
                print("✗", end=" ", flush=True)
        
        print()
        
        if len(mmd_list) > 0:
            result = {
                'dataset': dataset['name'],
                'method': method_name,
                'mmd_mean': np.mean(mmd_list),
                'mmd_std': np.std(mmd_list),
                'diversity_mean': np.mean(div_list),
                'diversity_std': np.std(div_list),
                'indistinguishability_mean': np.mean(inds_list),
                'indistinguishability_std': np.std(inds_list),
                'successful_runs': len(mmd_list),
            }
            
            print(f"    Results ({len(mmd_list)}/{N_REPEATS} runs):")
            print(f"      MMD:                  {result['mmd_mean']:.6f} ± {result['mmd_std']:.6f}")
            print(f"      Diversity:            {result['diversity_mean']:.6f} ± {result['diversity_std']:.6f}")
            print(f"      Indistinguishability: {result['indistinguishability_mean']:.6f} ± {result['indistinguishability_std']:.6f}")
            
            all_results.append(result)

# ==========================
# Export Results
# ==========================

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    
    # CSV
    csv_path = f'{OUTPUT_DIR}/comparison_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    
    # Visualizations
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        sns.barplot(data=results_df, x='method', y='mmd_mean', ax=axes[0], palette='Set2', errorbar=('sd', 1))
        axes[0].set_title('Maximum Mean Discrepancy\nLower is Better', fontweight='bold')
        
        sns.barplot(data=results_df, x='method', y='diversity_mean', ax=axes[1], palette='Set2', errorbar=('sd', 1))
        axes[1].set_title('Data Diversity\nHigher is Better', fontweight='bold')
        
        sns.barplot(data=results_df, x='method', y='indistinguishability_mean', ax=axes[2], palette='Set2', errorbar=('sd', 1))
        axes[2].set_title('Indistinguishability\nLower is Better', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/comparison_metrics.pdf', dpi=300)
        plt.close()
        print(f"✓ Saved: comparison_metrics.pdf")
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
else:
    print("ERROR: No results generated")

print("\n" + "="*80)
