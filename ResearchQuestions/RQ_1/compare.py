import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from ctgan import CTGAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.linalg import sqrtm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
from scipy.stats import f_oneway, kruskal

from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import time
from scipy.stats import ks_2samp


from BN_AUG_SDG import BN_AUG_SDG
from BGAN_SDG import BGAN_SDG

def plot_feature_distributions(real_data, bgan_data, bnaug_data, features=None, bins=30):
    if features is None:
        features = real_data.columns

    for feature in features:
        plt.figure(figsize=(10, 5))

        # Plot the distributions
        sns.histplot(real_data[feature], color='blue', label='Real', kde=True, stat="density", bins=bins, alpha=0.5)
        sns.histplot(bgan_data[feature], color='red', label='BGAN', kde=True, stat="density", bins=bins, alpha=0.5)
        sns.histplot(bnaug_data[feature], color='green', label='BN-AUG-SDG', kde=True, stat="density", bins=bins, alpha=0.5)

        # Perform KS tests
        ks_bgan = ks_2samp(real_data[feature], bgan_data[feature])
        ks_bnaug = ks_2samp(real_data[feature], bnaug_data[feature])

        # Annotate results
        plt.title(f'Distribution Comparison: {feature}\n'
                  f'KS p-value (BGAN): {ks_bgan.pvalue:.4e}, '
                  f'KS p-value (BN-AUG-SDG): {ks_bnaug.pvalue:.4e}')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_correlation_matrices(real_data, bgan_data, bnaug_data):
    real_data_numeric = real_data.select_dtypes(include=[np.number])
    bgan_data_numeric = bgan_data.select_dtypes(include=[np.number])
    bnaug_data_numeric = bnaug_data.select_dtypes(include=[np.number])
    
    corr_real = real_data_numeric.corr()
    corr_bgan = bgan_data_numeric.corr()
    corr_bnaug = bnaug_data_numeric.corr()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(corr_real, ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True)
    axes[0].set_title('Real Correlation')
    
    sns.heatmap(corr_bgan, ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True)
    axes[1].set_title('BGAN Correlation')
    
    sns.heatmap(corr_bnaug, ax=axes[2], cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True)
    axes[2].set_title('BN-AUG-SDG Correlation')
    
    plt.tight_layout()
    plt.show()

def plot_pca(real_data, bgan_data, bnaug_data):
    """Plot PCA visualization of real and synthetic data"""
    # One-hot encode all datasets
    real_data_encoded = pd.get_dummies(real_data)
    bgan_data_encoded = pd.get_dummies(bgan_data)
    bnaug_data_encoded = pd.get_dummies(bnaug_data)

    # Get union of all columns
    all_columns = sorted(set(real_data_encoded.columns) | 
                        set(bgan_data_encoded.columns) | 
                        set(bnaug_data_encoded.columns))
    
    # Reindex and fill NaN with 0
    real_data_encoded = real_data_encoded.reindex(columns=all_columns).fillna(0)
    bgan_data_encoded = bgan_data_encoded.reindex(columns=all_columns).fillna(0)
    bnaug_data_encoded = bnaug_data_encoded.reindex(columns=all_columns).fillna(0)

    # Combine data for PCA
    combined_data = pd.concat([real_data_encoded, bgan_data_encoded, bnaug_data_encoded])
    
    # Scale the data before PCA
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_data)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_scaled)

    # Split back into original datasets
    n_real = len(real_data)
    n_bgan = len(bgan_data)
    pca_real = pca_result[:n_real]
    pca_bgan = pca_result[n_real:n_real+n_bgan]
    pca_bnaug = pca_result[n_real+n_bgan:]

    # Plot
    plt.figure(figsize=(10,8))
    plt.scatter(pca_real[:,0], pca_real[:,1], alpha=0.5, label='Real', c='blue', s=30)
    plt.scatter(pca_bgan[:,0], pca_bgan[:,1], alpha=0.5, label='BGAN', c='red', s=30)
    plt.scatter(pca_bnaug[:,0], pca_bnaug[:,1], alpha=0.5, label='BN-AUG-SDG', c='green', s=30)
    
    # Add variance explained
    var_explained = pca.explained_variance_ratio_
    plt.xlabel(f'First PC ({var_explained[0]:.1%} variance explained)')
    plt.ylabel(f'Second PC ({var_explained[1]:.1%} variance explained)')
    
    plt.legend()
    plt.title('PCA projection: Real vs BGAN vs BN-AUG-SDG')
    plt.show()

def compute_mmd(X, Y, kernel_bandwidth=None):
    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    Y_std = scaler.transform(Y)
    # Median heuristic for bandwidth if not provided
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

def calculate_fid(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """Compute FID between two multivariate Gaussians"""
    # Product might be nearly singular
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Handle imaginary components due to numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_fid(real, synthetic):
    mu1 = np.mean(real, axis=0)
    mu2 = np.mean(synthetic, axis=0)
    sigma1 = np.cov(real, rowvar=False)
    sigma2 = np.cov(synthetic, rowvar=False)
    try:
        return calculate_fid(mu1, mu2, sigma1, sigma2)
    except Exception as e:
        print(f"FID Calculation Error: {e}")
        return np.nan

def calculate_diversity(data):
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    return score

def plot_uncertainty_heatmap(synthetic_data, title="Uncertainty Heatmap", columns=None):
    """Plot feature-wise variance as a heatmap for synthetic data."""
    if isinstance(synthetic_data, pd.DataFrame):
        data = synthetic_data.select_dtypes(include=[np.number]).values
        colnames = synthetic_data.select_dtypes(include=[np.number]).columns
    else:
        data = synthetic_data
        if columns is not None:
            colnames = columns
        else:
            colnames = [f"f{i}" for i in range(data.shape[1])]
    feature_variances = np.var(data, axis=0)
    plt.figure(figsize=(10, 1))
    sns.heatmap(feature_variances[np.newaxis, :], cmap="YlOrRd", cbar=True, xticklabels=colnames)
    plt.title(title)
    plt.yticks([])
    plt.show()

def classifier_performance(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

def train_and_sample_ctgan(real_train, discrete_columns):
        ctgan = CTGAN(epochs=1, discriminator_steps=5, batch_size = 200)
        ctgan.fit(real_train, discrete_columns)
        return ctgan.sample(len(real_train))

def plot_tsne(real_data, bgan_data, bnaug_data):
    all_data = np.vstack([real_data, bgan_data, bnaug_data])
    labels = np.array(['Real'] * len(real_data) + ['BGAN'] * len(bgan_data) + ['BN-AUG-SDG'] * len(bnaug_data))

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_result = tsne.fit_transform(all_data)

    plt.figure(figsize=(10,8))
    for label, color in zip(['Real', 'BGAN', 'BN-AUG-SDG'], ['blue', 'red', 'green']):
        idx = labels == label
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=label, alpha=0.5, s=30, c=color)
    plt.legend()
    plt.title('t-SNE projection: Real vs BGAN vs BN-AUG-SDG')
    plt.show()

def compute_uncertainty_metrics(scaled_data, feature_names=None, top_n=5, label=""):
    variances = np.var(scaled_data, axis=0)
    mean_uncertainty = np.mean(variances)
    
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(variances))]

    top_indices = np.argsort(-variances)[:top_n]
    top_features = [(feature_names[i], variances[i]) for i in top_indices]

    print(f"\n=== Uncertainty Metrics ({label}) ===")
    print(f"Mean Variance (Global Uncertainty): {mean_uncertainty:.6f}")
    print(f"Top {top_n} Most Uncertain Features:")
    for fname, var in top_features:
        print(f"  {fname}: {var:.6f}")

    return variances  # For plotting


def hyperparameter_search(
    real_train, real_eval, discrete_columns,
    bgan_param_grid, bnaug_param_grid, ctgan_param_grid, tvae_param_grid,
    gc_param_grid,
    n_samples=1000
):
    results = []

    # BGAN grid search
    for params in ParameterGrid(bgan_param_grid):
        print(f"Testing BGAN params: {params}")
        start = time.time()
        bgan = BGAN_SDG(**params)
        bgan.bgan.fit(real_train, discrete_columns)
        synthetic = bgan.bgan.sample(n_samples)
        metrics = evaluate_sdg_metrics(real_eval, synthetic)
        elapsed = time.time() - start
        results.append({'Method': 'BGAN', **params, **metrics ,'Runtime': elapsed})

    # BN-AUG-SDG grid search
    for params in ParameterGrid(bnaug_param_grid):
        print(f"Testing BN-AUG-SDG params: {params}")
        start = time.time()
        bnaug = BN_AUG_SDG(**params)
        bnaug.fit(real_train, discrete_columns)
        synthetic = bnaug.sample(n_samples)
        metrics = evaluate_sdg_metrics(real_eval, synthetic)
        elapsed = time.time() - start
        results.append({'Method': 'BN-AUG-SDG', **params, **metrics, 'Runtime': elapsed})

    # CTGAN grid search
    for params in ParameterGrid(ctgan_param_grid):
        print(f"Testing CTGAN params: {params}")
        start = time.time()
        ctgan = CTGAN(**params)
        ctgan.fit(real_train, discrete_columns)
        synthetic = ctgan.sample(n_samples)
        metrics = evaluate_sdg_metrics(real_eval, synthetic)
        elapsed = time.time() - start
        results.append({'Method': 'CTGAN', **params, **metrics, 'Runtime': elapsed})

    # TVAE grid search
    for params in ParameterGrid(tvae_param_grid):
        print(f"Testing TVAE params: {params}")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_train)
        start = time.time()
        tvae = TVAESynthesizer(metadata, **params)
        tvae.fit(real_train)
        synthetic = tvae.sample(n_samples)
        metrics = evaluate_sdg_metrics(real_eval, synthetic)
        elapsed = time.time() - start
        results.append({'Method': 'TVAE', **params, **metrics, 'Runtime': elapsed})

    # Gaussian Copula grid search
    for params in ParameterGrid(gc_param_grid):
        print(f"Testing Gaussian Copula params: {params}")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_train)
        start = time.time()
        gc = GaussianCopulaSynthesizer(metadata)
        gc.fit(real_train)
        synthetic = gc.sample(n_samples)
        metrics = evaluate_sdg_metrics(real_eval, synthetic)
        elapsed = time.time() - start
        results.append({'Method': 'Gaussian Copula', **params, **metrics, 'Runtime': elapsed})

    return pd.DataFrame(results)


def evaluate_sdg_metrics(real_eval, synthetic):
    # You can expand this to include all your metrics
    real_eval_enc = pd.get_dummies(real_eval)
    synthetic_enc = pd.get_dummies(synthetic)
    cols = sorted(set(real_eval_enc.columns) | set(synthetic_enc.columns))
    real_eval_enc = real_eval_enc.reindex(columns=cols, fill_value=0)
    synthetic_enc = synthetic_enc.reindex(columns=cols, fill_value=0)
    scaler = StandardScaler()
    real_eval_scaled = scaler.fit_transform(real_eval_enc)
    synthetic_scaled = scaler.transform(synthetic_enc)

    # Subsample for metrics
    def subsample(X, n=2000):
        if X.shape[0] > n:
            idx = np.random.choice(X.shape[0], n, replace=False)
            return X[idx]
        return X

    real_eval_scaled_metric = subsample(real_eval_scaled)
    synthetic_scaled_metric = subsample(synthetic_scaled)

    mmd = compute_mmd(real_eval_scaled_metric, synthetic_scaled_metric)
    diversity = calculate_diversity(synthetic_scaled_metric)
    acc = cross_val_score(
        LogisticRegressionCV(max_iter=1000, cv=3, random_state=42, n_jobs=1),
        np.vstack([real_eval_scaled_metric, synthetic_scaled_metric]),
        np.hstack([np.zeros(len(real_eval_scaled_metric)), np.ones(len(synthetic_scaled_metric))]),
        cv=3, scoring='accuracy', n_jobs=1
    ).mean()
    return {'MMD': mmd, 'Diversity': diversity, 'Indistinguishability': acc}

def evaluate_sdg(real_train, real_eval, bgan_data, bnaug_data, ctgan_data, bn_off_data, features=None):
    real_train_enc = pd.get_dummies(real_train)
    real_eval_enc = pd.get_dummies(real_eval)
    bgan_enc = pd.get_dummies(bgan_data)
    bnaug_enc = pd.get_dummies(bnaug_data)
    ctgan_enc = pd.get_dummies(ctgan_data)
    bn_off_enc = pd.get_dummies(bn_off_data)

    # Align columns
    cols = sorted(set(real_train_enc.columns) | set(real_eval_enc.columns) |
                  set(bgan_enc.columns) | set(bnaug_enc.columns) | set(ctgan_enc.columns))
    real_train_enc = real_train_enc.reindex(columns=cols, fill_value=0)
    real_eval_enc = real_eval_enc.reindex(columns=cols, fill_value=0)
    bgan_enc = bgan_enc.reindex(columns=cols, fill_value=0)
    bnaug_enc = bnaug_enc.reindex(columns=cols, fill_value=0)
    ctgan_enc = ctgan_enc.reindex(columns=cols, fill_value=0)
    bn_off_enc = bn_off_enc.reindex(columns=cols, fill_value=0)

    # Scale
    scaler = StandardScaler()
    real_eval_scaled = scaler.fit_transform(real_eval_enc)
    bgan_scaled = scaler.transform(bgan_enc)
    bnaug_scaled = scaler.transform(bnaug_enc)
    ctgan_scaled = scaler.transform(ctgan_enc)
    bn_off_scaled = scaler.transform(bn_off_enc)

    # Visualizations
    plot_feature_distributions(real_eval, bgan_data, bnaug_data, features)
    plot_correlation_matrices(real_eval, bgan_data, bnaug_data)
    plot_pca(real_eval, bgan_data, bnaug_data)
    plot_tsne(real_eval_scaled, bgan_scaled, bnaug_scaled)
    bgan_vars = compute_uncertainty_metrics(bgan_scaled, real_eval_enc.columns, label="BGAN")
    plot_uncertainty_heatmap(bgan_scaled, title="BGAN Uncertainty Heatmap", columns=real_eval_enc.columns)

    bnaug_vars = compute_uncertainty_metrics(bnaug_scaled, real_eval_enc.columns, label="BN-AUG-SDG")
    plot_uncertainty_heatmap(bnaug_scaled, title="BN-AUG-SDG Uncertainty Heatmap", columns=real_eval_enc.columns)

    ctgan_vars = compute_uncertainty_metrics(ctgan_scaled, real_eval_enc.columns, label="CTGAN")
    plot_uncertainty_heatmap(ctgan_scaled, title="CTGAN Uncertainty Heatmap", columns=real_eval_enc.columns)

    bn_off_vars = compute_uncertainty_metrics(bn_off_scaled, real_eval_enc.columns, label="BN-OFF-SDG")
    plot_uncertainty_heatmap(bn_off_scaled, title="BN-OFF-SDG Uncertainty Heatmap", columns=real_eval_enc.columns)


    # Subsample for metrics to avoid OOM
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

    # Metric results
    metrics_summary = []

    # MMD
    mmd_bgan = compute_mmd(real_eval_scaled_metric, bgan_scaled_metric)
    mmd_bnaug = compute_mmd(real_eval_scaled_metric, bnaug_scaled_metric)
    mmd_ctgan = compute_mmd(real_eval_scaled_metric, ctgan_scaled_metric)
    mmd_bn_off = compute_mmd(real_eval_scaled_metric, bn_off_scaled_metric)

    # FID
    #fid_bgan = get_fid(real_eval_scaled, bgan_scaled)
    #fid_bnaug = get_fid(real_eval_scaled, bnaug_scaled)
    #fid_ctgan = get_fid(real_eval_scaled, ctgan_scaled)


    # Diversity
    diversity_bgan = calculate_diversity(bgan_scaled_metric)
    diversity_bnaug = calculate_diversity(bnaug_scaled_metric)
    diversity_ctgan = calculate_diversity(ctgan_scaled_metric)
    diversity_bn_off = calculate_diversity(bn_off_scaled_metric)

    print("\n=== Similarity and Diversity Metrics ===")
    print(f"MMD (Real vs BGAN): {mmd_bgan:.5f}")
    print(f"MMD (Real vs BN-AUG-SDG): {mmd_bnaug:.5f}")
    print(f"MMD (Real vs CTGAN): {mmd_ctgan:.5f}")
    print(f"MMD (Real vs BN-OFF-SDG): {mmd_bn_off:.5f}")
    #print(f"FID (Real vs BGAN): {fid_bgan:.5f}")
    #print(f"FID (Real vs BN-AUG-SDG): {fid_bnaug:.5f}")
    #print(f"FID (Real vs CTGAN): {fid_ctgan:.5f}")
    print(f"Diversity (Silhouette) for BGAN: {diversity_bgan:.4f}")
    print(f"Diversity (Silhouette) for BN-AUG-SDG: {diversity_bnaug:.4f}")
    print(f"Diversity (Silhouette) for CTGAN: {diversity_ctgan:.4f}")
    print(f"Diversity (Silhouette) for BN-OFF-SDG: {diversity_bn_off:.4f}")

    # Subsample for classifier metrics to avoid OOM
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
    # If you add BN-only: bn_only_scaled_sub = subsample(bn_only_scaled)

    # Classifier Accuracy (Cross-Val)
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
            "MMD": compute_mmd(real_eval_scaled_sub, syn_scaled),
            "FID": get_fid(real_eval_scaled_sub, syn_scaled),
            "Diversity": calculate_diversity(syn_scaled),
            "Classifier Accuracy": acc_mean
        })

    # Summary Table
    print("\n=== Summary Table ===")
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.sort_values(by="Classifier Accuracy"))

def plot_uncertainty_delta(real_data, synthetic_data, title="Δ Uncertainty (Variance)", cols=None):
    """Compare variance between real and synthetic data."""
    if isinstance(real_data, pd.DataFrame):
        real = real_data.select_dtypes(include=[np.number])
        synth = synthetic_data[real.columns]  # Align columns
    else:
        real, synth = real_data, synthetic_data

    delta_var = np.var(synth.values, axis=0) - np.var(real.values, axis=0)

    colnames = cols if cols else real.columns
    plt.figure(figsize=(12, 1))
    sns.heatmap(delta_var[np.newaxis, :], cmap="coolwarm", center=0, xticklabels=colnames)
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_sample_entropy(probs, model_name="Model"):
    """Plot entropy distribution over synthetic samples."""
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    sns.histplot(entropy, kde=True, color='purple')
    plt.title(f"Predictive Entropy Distribution: {model_name}")
    plt.xlabel("Entropy")
    plt.ylabel("Sample Count")
    plt.show()
    return entropy

def correlate_uncertainty_with_sdg(real_df, synth_df, y_true, y_pred, feature_names=None):
    """Correlate per-feature variance with downstream task performance."""
    if isinstance(real_df, pd.DataFrame):
        real_df = real_df.select_dtypes(include=[np.number])
        synth_df = synth_df[real_df.columns]

    # Feature uncertainty = |synthetic variance - real variance|
    real_var = np.var(real_df.values, axis=0)
    synth_var = np.var(synth_df.values, axis=0)
    delta_var = np.abs(synth_var - real_var)

    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro-F1: {f1:.4f}")
    
    if feature_names is None:
        feature_names = real_df.columns

    # Plot
    plt.figure(figsize=(8, 4))
    sns.barplot(x=feature_names, y=delta_var)
    plt.title(f"Feature-wise Uncertainty vs F1={f1:.2f}")
    plt.xticks(rotation=45)
    plt.ylabel("|Δ Variance|")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load real data and preprocess
    real_data = pd.read_csv("http://ctgan-demo.s3.amazonaws.com/census.csv.gz")
    discrete_columns = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
    real_train, real_eval = train_test_split(real_data, test_size=0.3, random_state=42)

    #SAMPLING TO ESNURE TRAINING IS WORKING WELL, GET RID OF LATER!!
    real_train = real_train.sample(n=1000, random_state=42)
    real_eval = real_eval.sample(n=1000, random_state=42)

    # Generate synthetic data
    #vanilla_bgan = BGAN_SDG(epochs=25)
    #vanilla_bgan.bgan.fit(real_train, discrete_columns)
    #synthetic_vanilla = vanilla_bgan.bgan.sample(len(real_eval))

    #bn_bgan = BN_AUG_SDG(epochs=25, batch_norm=True, bn_influence = 1)
    #bn_bgan.fit(real_train, discrete_columns)
    #synthetic_bn = bn_bgan.sample(len(real_eval))

    #synthetic_ctgan = train_and_sample_ctgan(real_train, discrete_columns)

    #bn_off_bgan = BN_AUG_SDG(epochs=25, batch_norm=False)
    #bn_off_bgan.fit(real_train, discrete_columns)
    #synthetic_bn_off = bn_off_bgan.sample(len(real_eval))

    # TVAE
    # tvae = TVAESynthesizer(epochs=1)
    # tvae.fit(real_train)
    # synthetic_tvae = tvae.sample(len(real_eval))

    # GaussianCopula (BGMM)
    # gc = GaussianCopulaSynthesizer()
    # gc.fit(real_train)
    # synthetic_gc = gc.sample(len(real_eval))

    # Add noise to synthetic data
    #for df in [synthetic_vanilla, synthetic_bn, synthetic_ctgan]:
         #num_cols = df.select_dtypes(include=[np.number]).columns
         # Cast numeric columns to float before adding noise
         #df[num_cols] = df[num_cols].astype(float)
         #noise = np.random.normal(0, 0.01, size=df[num_cols].shape)
         # Ensure all numerical columns are float64 before adding noise
         #df.loc[:, df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).astype(np.float64)
         #df.loc[:, df.select_dtypes(include=[np.number]).columns] += noise

    #Evaluate using held-out real data
    #evaluate_sdg(real_train, real_eval, synthetic_vanilla, synthetic_bn, synthetic_ctgan, synthetic_bn_off)

    #plot_uncertainty_delta(real_eval, synthetic_bn, "BN-AUG: Δ Uncertainty")
    #plot_uncertainty_delta(real_eval, synthetic_vanilla, "Vanilla: Δ Uncertainty")

######################################################################################################
#HYPERPARAMETER SEARCH

    bgan_param_grid = {
        #'epochs': [1]
         #'embedding_dim': [128, 256],
         # Add more BGAN-specific params here
    }

    bnaug_param_grid = {
         'epochs': [50],
         'bn_influence': [0.01, 0.1, 1, 10, 100, 1000],
         #'batch_norm': [True, False]
         # Add more BN-AUG-SDG-specific params here
    }

    ctgan_param_grid = {
         #'epochs': [1]
         #'batch_size': [100, 150],
         # Add more CTGAN-specific params here
    }

    tvae_param_grid = {
        #'epochs': [1]
            # Add more TVAE-specific params here
    }

    gc_param_grid = {
        #'enforce_min_max_values': [True],
            # Add more GaussianCopula-specific params here
    }

    #search_results = hyperparameter_search(
    #    real_train, real_eval, discrete_columns,
    #    bgan_param_grid, bnaug_param_grid, ctgan_param_grid, tvae_param_grid, gc_param_grid,
    #    n_samples=1000
    #)
    #print(search_results.sort_values(by="Indistinguishability", ascending=True))

    # Compare uncertainty metrics for different bn_influence values
    n_repeats = 50  # or more for robustness

    bnaug_uncertainty = []
    for params in ParameterGrid(bnaug_param_grid):
        for seed in range(n_repeats):
            print(f"\nEvaluating uncertainty for BN-AUG-SDG with params: {params}, seed: {seed}")
            bnaug = BN_AUG_SDG(**params)
            bnaug.fit(real_train.sample(frac=1, random_state=seed), discrete_columns)
            synthetic = bnaug.sample(1000)
            # One-hot encode and align columns
            real_eval_enc = pd.get_dummies(real_eval)
            synthetic_enc = pd.get_dummies(synthetic)
            cols = sorted(set(real_eval_enc.columns) | set(synthetic_enc.columns))
            real_eval_enc = real_eval_enc.reindex(columns=cols, fill_value=0)
            synthetic_enc = synthetic_enc.reindex(columns=cols, fill_value=0)
            scaler = StandardScaler()
            real_eval_scaled = scaler.fit_transform(real_eval_enc)
            synthetic_scaled = scaler.transform(synthetic_enc)
            # Compute uncertainty (mean variance)
            variances = np.var(synthetic_scaled, axis=0)
            mean_uncertainty = np.mean(variances)
            bnaug_uncertainty.append({
                "bn_influence": params["bn_influence"],
                "mean_uncertainty": mean_uncertainty,
                "seed": seed
            })
            print(f"Mean variance (uncertainty): {mean_uncertainty:.6f}")

    # Convert to DataFrame and plot
    uncertainty_df = pd.DataFrame(bnaug_uncertainty)

    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=uncertainty_df,
        x="bn_influence",
        y="mean_uncertainty",
        color="skyblue"
    )
    sns.stripplot(
        data=uncertainty_df,
        x="bn_influence",
        y="mean_uncertainty",
        color="black",
        alpha=0.5,
        jitter=0.15,
        size=3
    )
    plt.xlabel("BN Influence")
    plt.ylabel("Mean Variance (Uncertainty)")
    plt.title("BN Influence vs. Uncertainty (BN-AUG-SDG)")
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    # Prepare data for statistical test
    groups = []
    for bn_inf in sorted(uncertainty_df["bn_influence"].unique()):
        vals = uncertainty_df.loc[uncertainty_df["bn_influence"] == bn_inf, "mean_uncertainty"].values
        print(f"bn_influence={bn_inf}: n={len(vals)}")
        groups.append(vals)

    # Use ANOVA if you have more than two groups and data is roughly normal
    if len(groups) > 2:
        stat, p = f_oneway(*groups)
        print(f"\nANOVA F-statistic: {stat:.4f}, p-value: {p:.4e}")
    else:
        # Use Kruskal-Wallis for non-parametric or two groups
        stat, p = kruskal(*groups)
        print(f"\nKruskal-Wallis H-statistic: {stat:.4f}, p-value: {p:.4e}")

    if p < 0.05:
        print("Result: Statistically significant difference in uncertainty across bn_influence values.")
    else:
        print("Result: No statistically significant difference in uncertainty across bn_influence values.")