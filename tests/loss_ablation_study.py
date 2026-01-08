"""
Loss Ablation Study for BN-AUG-SDG

Tests different loss configurations (KL+CL, KL, CL, Adversarial) to measure impact on synthetic data quality.
Evaluates on MMD, diversity, and indistinguishability metrics using the Cancer dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from scipy.stats import kruskal, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import after path modification
from bn_bgan.bn_bgan_sdg import BN_AUG_SDG


# ==============================================================================
# Utility Functions for Metrics (independent implementation)
# ==============================================================================

def compute_mmd(X, Y, kernel_bandwidth=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two datasets.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    Y_std = scaler.transform(Y)
    
    if kernel_bandwidth is None:
        sample = np.vstack([X_std, Y_std])
        dists = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1))
        kernel_bandwidth = np.median(dists)
        if kernel_bandwidth == 0:
            kernel_bandwidth = 1.0
    
    gamma = 1.0 / (2 * kernel_bandwidth ** 2)
    XX = rbf_kernel(X_std, X_std, gamma=gamma)
    YY = rbf_kernel(Y_std, Y_std, gamma=gamma)
    XY = rbf_kernel(X_std, Y_std, gamma=gamma)
    
    return XX.mean() + YY.mean() - 2 * XY.mean()


def calculate_diversity(data):
    """
    Calculate diversity using Silhouette Score after KMeans clustering.
    """
    kmeans = KMeans(n_clusters=min(5, len(data)), random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    return score


def calculate_indistinguishability(real_data, synthetic_data):
    """
    Calculate indistinguishability via cross-validation classifier accuracy.
    """
    X = np.vstack([real_data, synthetic_data])
    y = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])
    
    clf = LogisticRegressionCV(max_iter=1000, cv=5, random_state=42, n_jobs=1)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    return np.mean(scores), np.std(scores)



def get_loss_config_label(use_kl_loss, use_uncertainty_loss):
    """
    Generate human-readable label for loss configuration.
    """
    if use_kl_loss and use_uncertainty_loss:
        return "KL+CL"
    elif use_kl_loss:
        return "KL"
    elif use_uncertainty_loss:
        return "CL"
    else:
        return "Adversarial"


# ==============================================================================
# Statistical Significance Tests
# ==============================================================================

def perform_kruskal_wallis_test(results_df, metric='mmd'):
    """
    Perform Kruskal-Wallis H-test for non-parametric comparison of multiple groups.
    Tests if there are statistically significant differences between loss configurations.
    
    Returns: (statistic, p_value, interpretation)
    """
    if metric not in results_df.columns:
        return None, None, f"Column '{metric}' not found in results"
    
    groups = [results_df[results_df['config'] == config][metric].values 
              for config in results_df['config'].unique()]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return None, None, "Insufficient groups for test"
    
    stat, p_value = kruskal(*groups)
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        interpretation = f"SIGNIFICANT (p={p_value:.4f} < {alpha})"
    else:
        interpretation = f"NOT SIGNIFICANT (p={p_value:.4f} >= {alpha})"
    
    return stat, p_value, interpretation


def perform_anova_test(results_df, metric='mmd'):
    """
    Perform one-way ANOVA for parametric comparison of multiple groups.
    
    Returns: (F-statistic, p_value, interpretation)
    """
    if metric not in results_df.columns:
        return None, None, f"Column '{metric}' not found in results"
    
    groups = [results_df[results_df['config'] == config][metric].values 
              for config in results_df['config'].unique()]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return None, None, "Insufficient groups for test"
    
    f_stat, p_value = f_oneway(*groups)
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        interpretation = f"SIGNIFICANT (p={p_value:.4f} < {alpha})"
    else:
        interpretation = f"NOT SIGNIFICANT (p={p_value:.4f} >= {alpha})"
    
    return f_stat, p_value, interpretation


def print_statistical_summary(results_df):
    """
    Print comprehensive statistical analysis of loss ablation results.
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)
    
    # MMD Analysis
    print("\n" + "-"*80)
    print("1. MAXIMUM MEAN DISCREPANCY (MMD) - Kruskal-Wallis Test")
    print("-"*80)
    print("   Null Hypothesis: All loss configurations produce equally similar synthetic data")
    print("   Alternative: At least one configuration differs significantly")
    
    h_stat, p_val, interp = perform_kruskal_wallis_test(results_df, 'mmd')
    if h_stat is not None:
        print(f"\n   H-statistic: {h_stat:.6f}")
        print(f"   p-value:     {p_val:.6f}")
        print(f"   Result:      {interp}")
    else:
        print(f"   {interp}")
    
    # Diversity Analysis
    print("\n" + "-"*80)
    print("2. DATA DIVERSITY - Kruskal-Wallis Test")
    print("-"*80)
    print("   Null Hypothesis: All loss configurations produce equally diverse synthetic data")
    print("   Alternative: At least one configuration differs significantly")
    
    h_stat, p_val, interp = perform_kruskal_wallis_test(results_df, 'diversity')
    if h_stat is not None:
        print(f"\n   H-statistic: {h_stat:.6f}")
        print(f"   p-value:     {p_val:.6f}")
        print(f"   Result:      {interp}")
    else:
        print(f"   {interp}")
    
    # Descriptive Statistics
    print("\n" + "-"*80)
    print("3. DESCRIPTIVE STATISTICS BY CONFIGURATION")
    print("-"*80)
    
    for config in results_df['config'].unique():
        config_data = results_df[results_df['config'] == config]
        print(f"\n   {config}:")
        if 'mmd' in config_data.columns:
            print(f"     MMD:       {config_data['mmd'].values[0]:.6f}")
        if 'diversity' in config_data.columns:
            print(f"     Diversity: {config_data['diversity'].values[0]:.6f}")
        if 'indistinguishability' in config_data.columns:
            print(f"     Indistinguishability: {config_data['indistinguishability'].values[0]:.6f}")
    
    print("\n" + "="*80)
# ==============================================================================

def load_dataset(dataset_name):
    """
    Load dataset from ARFF file or new_datasets folder (CSV, pickle, etc.).
    Handles both 'dataset_name' and 'folder/dataset_name' formats.
    """
    import pickle
    
    # Clean up path - remove folder prefix if present
    if '/' in dataset_name:
        clean_name = dataset_name.split('/')[-1]
    else:
        clean_name = dataset_name
    
    # Try datasets folder first (ARFF files)
    dataset_path = f'datasets/{clean_name}.arff'
    if os.path.exists(dataset_path):
        data, meta = arff.loadarff(dataset_path)
        data_df = pd.DataFrame(data)
        
        # Decode bytes columns if necessary
        for col in data_df.select_dtypes([object]):
            data_df[col] = data_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    else:
        # Try new_datasets folder (CSV, pickle, etc.)
        # Try different formats
        csv_path = f'new_datasets/{clean_name}.csv'
        pkl_path = f'new_datasets/{clean_name}.pkl'
        pickle_path = f'new_datasets/{clean_name}.pickle'
        file_no_ext = f'new_datasets/{clean_name}'
        
        if os.path.exists(csv_path):
            data_df = pd.read_csv(csv_path)
        elif os.path.exists(pkl_path):
            data_df = pd.read_pickle(pkl_path)
        elif os.path.exists(pickle_path):
            data_df = pd.read_pickle(pickle_path)
        elif os.path.isfile(file_no_ext):
            # Try loading as pickle without extension (check if it's a file first)
            try:
                data_df = pd.read_pickle(file_no_ext)
            except:
                raise FileNotFoundError(f"Could not load dataset: {clean_name} (tried pickle format)")
        else:
            raise FileNotFoundError(f"Dataset not found: {clean_name}")
    
    discrete_columns = data_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return data_df, discrete_columns


def preprocess_for_metrics(real_data, synthetic_data):
    """
    Preprocess data for metric computation (one-hot encoding, scaling).
    """
    # One-hot encode
    real_enc = pd.get_dummies(real_data)
    synth_enc = pd.get_dummies(synthetic_data)
    
    # Align columns
    cols = sorted(set(real_enc.columns) | set(synth_enc.columns))
    real_enc = real_enc.reindex(columns=cols, fill_value=0)
    synth_enc = synth_enc.reindex(columns=cols, fill_value=0)
    
    # Scale
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_enc)
    synth_scaled = scaler.transform(synth_enc)
    
    return real_scaled, synth_scaled


# ==============================================================================
# Loss Ablation Study
# ==============================================================================

def evaluate_loss_config(real_train, real_eval, discrete_columns, use_kl_loss, use_uncertainty_loss, epochs=5):
    """
    Evaluate BN-AUG-SDG with a specific loss configuration.
    """
    config_label = get_loss_config_label(use_kl_loss, use_uncertainty_loss)
    
    print(f"\n  Testing {config_label}...", end="", flush=True)
    
    try:
        # Remove any null values that could cause issues
        real_train_clean = real_train.dropna()
        real_eval_clean = real_eval.dropna()
        
        if len(real_train_clean) == 0 or len(real_eval_clean) == 0:
            print(f" ✗ FAILED: No data after removing nulls")
            return None
        
        # Train model with specified loss configuration
        model = BN_AUG_SDG(
            epochs=epochs,
            batch_norm=True,
            use_kl_loss=use_kl_loss,
            use_uncertainty_loss=use_uncertainty_loss,
            optimizer_type='adam',
            bn_influence=0.5
        )
        model.fit(real_train_clean, discrete_columns)
        
        # Generate synthetic data
        synthetic_data = model.sample(len(real_eval_clean))
        
        # Preprocess for metrics
        real_eval_scaled, synth_scaled = preprocess_for_metrics(real_eval_clean, synthetic_data)
        
        # Compute metrics
        mmd = compute_mmd(real_eval_scaled, synth_scaled)
        diversity = calculate_diversity(synth_scaled)
        inds_mean, inds_std = calculate_indistinguishability(real_eval_scaled, synth_scaled)
        
        print(f" ✓ MMD={mmd:.4f}, Div={diversity:.4f}, Inds={inds_mean:.4f}")
        
        return {
            'config': config_label,
            'use_kl_loss': use_kl_loss,
            'use_uncertainty_loss': use_uncertainty_loss,
            'mmd': mmd,
            'diversity': diversity,
            'indistinguishability': inds_mean,
            'indistinguishability_std': inds_std
        }
    
    except Exception as e:
        print(f" ✗ FAILED: {str(e)}")
        return None


def run_loss_ablation(real_train, real_eval, discrete_columns, epochs=5, n_runs=10):
    """
    Run full loss ablation study with all loss configuration combinations.
    Multiple runs per configuration for robust statistics.
    """
    results = []
    
    # Test all 4 loss configuration combinations
    loss_configs = [
        (True, True),    # KL+CL
        (True, False),   # KL
        (False, True),   # CL
        (False, False),  # Adversarial
    ]
    
    print("\n" + "="*70)
    print(f"LOSS ABLATION STUDY - CANCER DATASET ({n_runs} runs per configuration)")
    print("="*70)
    
    for use_kl, use_uncertainty in loss_configs:
        config_label = get_loss_config_label(use_kl, use_uncertainty)
        print(f"\n{config_label} (running {n_runs} iterations):")
        
        mmd_scores = []
        diversity_scores = []
        inds_scores = []
        
        for run_idx in range(n_runs):
            try:
                model = BN_AUG_SDG(
                    epochs=epochs,
                    batch_norm=True,
                    use_kl_loss=use_kl,
                    use_uncertainty_loss=use_uncertainty,
                    optimizer_type='adam',
                    bn_influence=0.5
                )
                model.fit(real_train, discrete_columns)
                synthetic_data = model.sample(len(real_eval))
                
                # Compute metrics
                real_eval_scaled, synth_scaled = preprocess_for_metrics(real_eval, synthetic_data)
                mmd = compute_mmd(real_eval_scaled, synth_scaled)
                diversity = calculate_diversity(synth_scaled)
                inds_mean, inds_std = calculate_indistinguishability(real_eval_scaled, synth_scaled)
                
                mmd_scores.append(mmd)
                diversity_scores.append(diversity)
                inds_scores.append(inds_mean)
                
                print(".", end="", flush=True)
            
            except Exception as e:
                print("✗", end="", flush=True)
        
        print(" ✓")
        
        if mmd_scores:
            result = {
                'config': config_label,
                'use_kl_loss': use_kl,
                'use_uncertainty_loss': use_uncertainty,
                'mmd': np.mean(mmd_scores),
                'mmd_std': np.std(mmd_scores),
                'diversity': np.mean(diversity_scores),
                'diversity_std': np.std(diversity_scores),
                'indistinguishability': np.mean(inds_scores),
                'indistinguishability_std': np.std(inds_scores),
                'runs': len(mmd_scores)
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df


# ==============================================================================
# Visualization and Output
# ==============================================================================

def create_loss_ablation_plots(results_df, output_dir='outputs'):
    """
    Create publication-ready plots for loss ablation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df.empty or 'config' not in results_df.columns:
        print("⚠ No results to plot (empty or invalid dataframe)")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # Plot 1: Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    metrics = ['mmd', 'diversity', 'indistinguishability']
    metric_labels = ['MMD', 'Diversity', 'Indistinguishability']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric not in results_df.columns:
            continue
            
        ax = axes[idx]
        bars = ax.bar(results_df['config'], results_df[metric], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results_df)])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'Loss Configurations: {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=11)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_ablation_metrics.pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/loss_ablation_metrics.pdf")
    plt.close()


def export_results(results_df, output_dir='outputs'):
    """
    Export results to CSV and LaTeX formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df.empty:
        print("⚠ No results to export (empty dataframe)")
        return
    
    # CSV export
    csv_path = f'{output_dir}/loss_ablation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # LaTeX export - only if we have the expected columns
    if 'config' in results_df.columns and 'mmd' in results_df.columns:
        latex_path = f'{output_dir}/loss_ablation_results.tex'
        latex_table = results_df[['config', 'mmd', 'diversity', 'indistinguishability']].copy()
        latex_table.columns = ['Configuration', 'MMD', 'Diversity', 'Indistinguishability']
        
        with open(latex_path, 'w') as f:
            f.write(latex_table.to_latex(index=False, float_format=lambda x: f'{x:.4f}'))
        
        print(f"✓ Saved: {latex_path}")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution block for loss ablation study.
    """
    
    # Configuration
    DATASET_NAME = 'Fetal_dataset'
    EPOCHS = 50
    TEST_SIZE = 0.3
    RANDOM_SEED = 42
    
    print("\n" + "="*70)
    print("LOSS ABLATION STUDY FOR BN-AUG-SDG")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading {DATASET_NAME}...")
    data, discrete_columns = load_dataset(DATASET_NAME)
    
    # Remove null values
    data = data.dropna()
    print(f"  Total samples after removing nulls: {len(data)}")
    
    # Split data
    real_train, real_eval = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    # For testing with smaller sample sizes (comment out for full dataset)
    real_train = real_train.sample(n=min(50, len(real_train)), random_state=RANDOM_SEED)
    real_eval = real_eval.sample(n=min(50, len(real_eval)), random_state=RANDOM_SEED)
    
    print(f"  Training samples: {len(real_train)}")
    print(f"  Evaluation samples: {len(real_eval)}")
    print(f"  Discrete columns: {discrete_columns}")
    
    # Run ablation study with 10 runs per configuration for robust statistics
    results = run_loss_ablation(real_train, real_eval, discrete_columns, epochs=EPOCHS, n_runs=10)
    
    # Display results
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    if not results.empty:
        print(results.to_string(index=False))
        
        # Perform statistical significance tests
        print_statistical_summary(results)
    else:
        print("No successful results to display")
    
    # Export and visualize
    export_results(results)
    create_loss_ablation_plots(results)
    
    print("\n" + "="*70)
    print("Loss ablation study complete!")
    print("="*70 + "\n")
