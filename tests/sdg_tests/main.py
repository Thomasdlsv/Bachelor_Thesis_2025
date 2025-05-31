import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.io import arff
import glob
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from tests.sdg_tests.sdg_visualizer import SDGVisualizer
from bn_bgan.bn_bgan_sdg import BN_AUG_SDG
from bgan.utility.bgan_sdg import BGAN_SDG


def evaluate_sdg(real_train, real_eval, features=None, epochs=1):
    """
    Evaluate and compare SDG models on real and synthetic data.
    Outputs similarity/diversity metrics and classifier accuracy.
    """
    vanilla_bgan = BGAN_SDG(epochs=epochs)
    vanilla_bgan.bgan.fit(real_train, discrete_columns)
    synthetic_vanilla = vanilla_bgan.bgan.sample(len(real_eval))

    # can adjust bn_influence to control the influence of bayesian network structure
    bn_bgan = BN_AUG_SDG(epochs=epochs, batch_norm=True, bn_influence = 0.5)
    bn_bgan.fit(real_train, discrete_columns)
    synthetic_bn = bn_bgan.sample(len(real_eval))

    # ==========================
    # VISUALIZE THE DAG STRUCUTRE 
    # ==========================
    #(COMMENT OUT TO SAVE TIME WHEN RUNNING MULTIPLE SIMULATIONS)
    bn_bgan.plot_bn_structure(weighted=True)

    synthetic_ctgan = SDGVisualizer.train_and_sample_ctgan(real_train, discrete_columns)

    bn_off_bgan = BN_AUG_SDG(epochs=EPOCHS, batch_norm=False)
    bn_off_bgan.fit(real_train, discrete_columns)
    synthetic_bn_off = bn_off_bgan.sample(len(real_eval))

    real_train_enc = pd.get_dummies(real_train)
    real_eval_enc = pd.get_dummies(real_eval)
    bgan_enc = pd.get_dummies(synthetic_vanilla)
    bnaug_enc = pd.get_dummies(synthetic_bn)
    ctgan_enc = pd.get_dummies(synthetic_ctgan)
    bn_off_enc = pd.get_dummies(synthetic_bn_off)

    # need to align the columns so that all datasets have the same features
    cols = sorted(set(real_train_enc.columns) | set(real_eval_enc.columns) |
                  set(bgan_enc.columns) | set(bnaug_enc.columns) | set(ctgan_enc.columns))
    real_train_enc = real_train_enc.reindex(columns=cols, fill_value=0)
    real_eval_enc = real_eval_enc.reindex(columns=cols, fill_value=0)
    bgan_enc = bgan_enc.reindex(columns=cols, fill_value=0)
    bnaug_enc = bnaug_enc.reindex(columns=cols, fill_value=0)
    ctgan_enc = ctgan_enc.reindex(columns=cols, fill_value=0)
    bn_off_enc = bn_off_enc.reindex(columns=cols, fill_value=0)

    # we scale with respect to the real evaluation set so that the synthetic data is comparable
    scaler = StandardScaler()
    real_eval_scaled = scaler.fit_transform(real_eval_enc)
    bgan_scaled = scaler.transform(bgan_enc)
    bnaug_scaled = scaler.transform(bnaug_enc)
    ctgan_scaled = scaler.transform(ctgan_enc)
    bn_off_scaled = scaler.transform(bn_off_enc)

    # we plot the scaled data for comparison
    SDGVisualizer.plot_feature_distributions(real_eval, synthetic_vanilla, synthetic_bn, features)
    SDGVisualizer.plot_correlation_matrices(real_eval, synthetic_vanilla, synthetic_bn)
    SDGVisualizer.plot_pca(real_eval, synthetic_vanilla, synthetic_bn)
    SDGVisualizer.plot_tsne(real_eval_scaled, bgan_scaled, bnaug_scaled)
    SDGVisualizer.plot_uncertainty_heatmap(bgan_scaled, title="BGAN Uncertainty Heatmap", columns=real_eval_enc.columns)
    SDGVisualizer.plot_uncertainty_heatmap(bnaug_scaled, title="BN-AUG-SDG Uncertainty Heatmap", columns=real_eval_enc.columns)
    SDGVisualizer.plot_uncertainty_heatmap(ctgan_scaled, title="CTGAN Uncertainty Heatmap", columns=real_eval_enc.columns)
    SDGVisualizer.plot_uncertainty_heatmap(bn_off_scaled, title="BN-OFF-SDG Uncertainty Heatmap", columns=real_eval_enc.columns)


    # Subsample for metrics to avoid overwhelming computations
    max_metric_samples = 2000
    def subsample(X, n=max_metric_samples):
        if X.shape[0] > n:
            idx = np.random.choice(X.shape[0], n, replace=False)
            return X[idx]
        return X

    real_eval_scaled_metric = subsample(real_eval_scaled)
    bgan_scaled_metric = subsample(bgan_scaled)
    bnaug_scaled_metric = subsample(bnaug_scaled)
    ctgan_scaled_metric = subsample(ctgan_scaled)
    bn_off_scaled_metric = subsample(bn_off_scaled)

    metrics_summary = []

    # Metric 1: MMD
    mmd_bgan = SDGVisualizer.compute_mmd(real_eval_scaled_metric, bgan_scaled_metric)
    mmd_bnaug = SDGVisualizer.compute_mmd(real_eval_scaled_metric, bnaug_scaled_metric)
    mmd_ctgan = SDGVisualizer.compute_mmd(real_eval_scaled_metric, ctgan_scaled_metric)
    mmd_bn_off = SDGVisualizer.compute_mmd(real_eval_scaled_metric, bn_off_scaled_metric)

    # Metric 2: Diversity
    diversity_bgan = SDGVisualizer.calculate_diversity(bgan_scaled_metric)
    diversity_bnaug = SDGVisualizer.calculate_diversity(bnaug_scaled_metric)
    diversity_ctgan = SDGVisualizer.calculate_diversity(ctgan_scaled_metric)
    diversity_bn_off = SDGVisualizer.calculate_diversity(bn_off_scaled_metric)

    print("\n=== Similarity and Diversity Metrics ===")
    print(f"MMD (Real vs BGAN): {mmd_bgan:.5f}")
    print(f"MMD (Real vs BN-AUG-SDG): {mmd_bnaug:.5f}")
    print(f"MMD (Real vs CTGAN): {mmd_ctgan:.5f}")
    print(f"MMD (Real vs BN-OFF-SDG): {mmd_bn_off:.5f}")
    print(f"Diversity (Silhouette) for BGAN: {diversity_bgan:.4f}")
    print(f"Diversity (Silhouette) for BN-AUG-SDG: {diversity_bnaug:.4f}")
    print(f"Diversity (Silhouette) for CTGAN: {diversity_ctgan:.4f}")
    print(f"Diversity (Silhouette) for BN-OFF-SDG: {diversity_bn_off:.4f}")

    # Subsample for classifier metrics to avoid overwhelming computations
    def subsample(X, n=2000):
        if X.shape[0] > n:
            idx = np.random.choice(X.shape[0], n, replace=False)
            return X[idx]
        return X

    real_eval_scaled_sub = subsample(real_eval_scaled)
    bgan_scaled_sub = subsample(bgan_scaled)
    bnaug_scaled_sub = subsample(bnaug_scaled)
    ctgan_scaled_sub = subsample(ctgan_scaled)
    bn_off_scaled_sub = subsample(bn_off_scaled)

    # Metric 3: Indistinguishability (Cross-Val)
    print("\n=== Classifier Accuracy Comparison ===")
    for label, syn_scaled in zip(["BGAN", "BN-AUG-SDG", "CTGAN", "BN-OFF-SDG"], [bgan_scaled_sub, bnaug_scaled_sub, ctgan_scaled_sub, bn_off_scaled_sub]):
        X = np.vstack([real_eval_scaled_sub, syn_scaled])
        y = np.hstack([np.zeros(len(real_eval_scaled_sub)), np.ones(len(syn_scaled))])
        clf = LogisticRegressionCV(max_iter=1000, cv=5, random_state=42, n_jobs=1)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        acc_mean, acc_std = np.mean(scores), np.std(scores)
        print(f"{label} Classifier Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        metrics_summary.append({
            "Method": label,
            "MMD": SDGVisualizer.compute_mmd(real_eval_scaled_sub, syn_scaled),
            "Diversity": SDGVisualizer.calculate_diversity(syn_scaled),
            "Classifier Accuracy": acc_mean
        })

    # Summarize all data in a table in the terminal
    print("\n=== Summary Table ===")
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.sort_values(by="Classifier Accuracy"))

# ==============================================================================================================================
# Main execution block to run the evaluation regarding the synthetic data generated of the model with respct to baseline methods.
# ==============================================================================================================================

if __name__ == "__main__":
    """
    Main execution block for benchmarking synthetic data generation methods on the Fetal_Dataset.arff dataset.
    This script:
        - Loads and preprocesses the dataset.
        - Defines and configures several synthetic data generation methods.
        - Runs hyperparameter search for each method.
        - Evaluates and visualizes model performance on real and synthetic data.
    """

    # relative path to the datasets folder
    data, meta = arff.loadarff('datasets/Fetal_Dataset.arff')
    real_data = pd.DataFrame(data)
    # decode bytes columns if necessary, meaning, if the columns are of type object we try to get the string representation
    # This is necessary for datasets that have byte strings
    # (e.g., categorical variables that are encoded as bytes)
    for col in real_data.select_dtypes([object]):
        real_data[col] = real_data[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    discrete_columns = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
    real_train, real_eval = train_test_split(real_data, test_size=0.3, random_state=42)

    # =============================================================================
    # FOR TESTING PURPOSES ONLY:
    # Comment the following lines to get the appropriate dataset size for testing
    # =============================================================================
    real_train = real_train.sample(n=10, random_state=42)
    real_eval = real_eval.sample(n=10, random_state=42)

    # ==================
    # TESTING PARAMETERS
    # ==================
    MAX_RUNS = 3
    N_SAMPLES = 1000 # refers to the number of samples to generate for each method during hyperparameter search
    EPOCHS = 1
  
    #Evaluate using held-out real data
    # This will output a lot of graphs with respects to the features, comment out if just running hyperparameter search
    print("\n=== Visualizing Model Performance ===")
    evaluate_sdg(real_train, real_eval, features=real_data.columns, epochs = EPOCHS)

    # ====================
    #HYPERPARAMETER SEARCH
    # ====================

    bgan_param_grid = {
        'epochs': [1],
        'use_uncertainty_loss': [True, False],
        'use_kl_loss': [True, False],   
    }

    bnaug_param_grid = {
         'epochs': [1],
         #'batch_norm': [True, False],
         'use_uncertainty_loss': [True, False],
         'use_kl_loss': [True, False],
         'optimizer_type': ["adam", "adamw", "rmsprop"], 
         #'bn_influence': [0.1, 0.5, 0.9]
    }

    ctgan_param_grid = {
         'epochs': [1],
    }

    tvae_param_grid = {
        'epochs': [1]
    }

    gc_param_grid = {
        'enforce_min_max_values': [True],
    }

    search_results = SDGVisualizer.hyperparameter_search(
        real_train, real_eval, discrete_columns,
        bgan_param_grid, bnaug_param_grid, ctgan_param_grid, tvae_param_grid, gc_param_grid, 
        max_runs=MAX_RUNS,
        n_samples=N_SAMPLES
    )
    print("\n=== Hyperparameter Search Results ===")
    print(search_results.to_string(index=False))

    print("\n=== Hyperparameter Search Results (MMD & Diversity) ===")
    print(
        search_results[[
            "Method", "epochs", "optimizer_type", "MMD_mean", "MMD_std", "Diversity_mean", "Diversity_std", "Runs"
        ]].to_string(index=False)
    )
