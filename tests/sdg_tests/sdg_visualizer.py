import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics.pairwise import rbf_kernel
from ctgan import CTGAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from sdv.metadata import SingleTableMetadata
from scipy.stats import ks_2samp

# import all models included for evaluation
from ctgan import CTGAN
from bgan.utility.bgan_sdg import BGAN_SDG
from bn_bgan.bn_bgan_sdg import BN_AUG_SDG
from sdv.single_table import TVAESynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

class SDGVisualizer:

    """
    SDGVisualizer provides a suite of static and class methods for evaluating and visualizing
    synthetic data generators (SDGs) such as BGAN, BN-AUG-SDG, CTGAN.

    Main functionalities include:
    - Visualization of feature distributions, correlation matrices, PCA and t-SNE projections
      to compare real and synthetic datasets.
    - Calculation of similarity and diversity metrics such as Maximum Mean Discrepancy (MMD)
      and Silhouette Score for diversity.
    - Visualization and analysis of uncertainty in synthetic data, including feature-wise
      variance heatmaps and delta-uncertainty plots.
    - Evaluation of classifier-based distinguishability between real and synthetic data.
    - Hyperparameter search utilities for benchmarking SDG models across parameter grids.
    - Utility functions for statistical tests, entropy analysis, and correlation of uncertainty
      with downstream task performance.

    Methods are designed to be modular and reusable for benchmarking, reporting, and
    research on synthetic data quality and utility.
    """

    @staticmethod
    def get_loss_config_label(use_kl_loss, use_uncertainty_loss):
        """
        Generate a human-readable label for loss configurations.
        
        Mapping:
        - Both True: "KL+CL" (KL Divergence + Calibration Loss)
        - KL True, CL False: "KL" 
        - KL False, CL True: "CL"
        - Both False: "Adversarial"
        
        Args:
            use_kl_loss: Boolean indicating if KL loss is used
            use_uncertainty_loss: Boolean indicating if Calibration/Uncertainty loss is used
            
        Returns:
            String label for the loss configuration
        """
        if use_kl_loss and use_uncertainty_loss:
            return "KL+CL"
        elif use_kl_loss:
            return "KL"
        elif use_uncertainty_loss:
            return "CL"
        else:
            return "Adversarial Loss Only"

    @staticmethod

    def plot_correlation_matrices(real_data, bgan_data, bnaug_data):

        """
        Plot correlation matrices for real, BGAN, and BN-AUG-SDG data side by side.
        """

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

    @staticmethod
    def plot_pca(real_data, bgan_data, bnaug_data):

        """
        Plot PCA projection (2D) of real, BGAN, and BN-AUG-SDG data.
        """
         
        real_data_encoded = pd.get_dummies(real_data)
        bgan_data_encoded = pd.get_dummies(bgan_data)
        bnaug_data_encoded = pd.get_dummies(bnaug_data)
        all_columns = sorted(set(real_data_encoded.columns) | set(bgan_data_encoded.columns) | set(bnaug_data_encoded.columns))
        real_data_encoded = real_data_encoded.reindex(columns=all_columns).fillna(0)
        bgan_data_encoded = bgan_data_encoded.reindex(columns=all_columns).fillna(0)
        bnaug_data_encoded = bnaug_data_encoded.reindex(columns=all_columns).fillna(0)
        combined_data = pd.concat([real_data_encoded, bgan_data_encoded, bnaug_data_encoded])
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_scaled)
        n_real = len(real_data)
        n_bgan = len(bgan_data)
        pca_real = pca_result[:n_real]
        pca_bgan = pca_result[n_real:n_real+n_bgan]
        pca_bnaug = pca_result[n_real+n_bgan:]
        plt.figure(figsize=(10,8))
        plt.scatter(pca_real[:,0], pca_real[:,1], alpha=0.5, label='Real', c='blue', s=30)
        plt.scatter(pca_bgan[:,0], pca_bgan[:,1], alpha=0.5, label='BGAN', c='red', s=30)
        plt.scatter(pca_bnaug[:,0], pca_bnaug[:,1], alpha=0.5, label='BN-AUG-SDG', c='green', s=30)
        var_explained = pca.explained_variance_ratio_
        plt.xlabel(f'First PC ({var_explained[0]:.1%} variance explained)')
        plt.ylabel(f'Second PC ({var_explained[1]:.1%} variance explained)')
        plt.legend()
        plt.title('PCA projection: Real vs BGAN vs BN-AUG-SDG')
        plt.show()

    @staticmethod
    def plot_tsne(real_data, bgan_data, bnaug_data):

        """
        Plot t-SNE projection (2D) of real, BGAN, and BN-AUG-SDG data.
        """

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

    @staticmethod
    def plot_uncertainty_heatmap(synthetic_data, title="Uncertainty Heatmap", columns=None):

        """
        Plot feature-wise variance as a heatmap for synthetic data.
        """

        if isinstance(synthetic_data, np.ndarray):
            data = synthetic_data
            colnames = columns if columns is not None else [f"f{i}" for i in range(data.shape[1])]
        else:
            data = synthetic_data.select_dtypes(include=[np.number]).values
            colnames = synthetic_data.select_dtypes(include=[np.number]).columns
        feature_variances = np.var(data, axis=0)
        plt.figure(figsize=(10, 1))
        sns.heatmap(feature_variances[np.newaxis, :], cmap="YlOrRd", cbar=True, xticklabels=colnames)
        plt.title(title)
        plt.yticks([])
        plt.show()

    @staticmethod
    def plot_uncertainty_delta(real_data, synthetic_data, title="Δ Uncertainty (Variance)", cols=None):

        """
        Plot the difference in feature-wise variance between real and synthetic data.
        """

        if isinstance(real_data, np.ndarray):
            real = real_data
            synth = synthetic_data
            colnames = cols if cols else [f"f{i}" for i in range(real.shape[1])]
        else:
            real = real_data.select_dtypes(include=[np.number])
            synth = synthetic_data[real.columns]
            colnames = cols if cols else real.columns
        delta_var = np.var(synth.values, axis=0) - np.var(real.values, axis=0)
        plt.figure(figsize=(12, 1))
        sns.heatmap(delta_var[np.newaxis, :], cmap="coolwarm", center=0, xticklabels=colnames)
        plt.title(title)
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_aggregated_uncertainty(uncertainty_df):

        """
        Plot boxplots and stripplots of mean uncertainty grouped by BN influence.
        """

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

    @staticmethod
    def plot_feature_distributions(real_data, bgan_data, bnaug_data, features=None, bins=30):

        """
        Plot and compare feature distributions for real, BGAN, and BN-AUG-SDG data.
        Performs KS tests and annotates p-values.
        """

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

    @staticmethod
    def plot_correlation_matrices(real_data, bgan_data, bnaug_data):

        """
        Plot correlation matrices for real, BGAN, and BN-AUG-SDG data side by side.
        """

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

    @staticmethod
    def plot_pca(real_data, bgan_data, bnaug_data):

        """Plot PCA visualization of real and synthetic data"""

        real_data_encoded = pd.get_dummies(real_data)
        bgan_data_encoded = pd.get_dummies(bgan_data)
        bnaug_data_encoded = pd.get_dummies(bnaug_data)

        all_columns = sorted(set(real_data_encoded.columns) | 
                            set(bgan_data_encoded.columns) | 
                            set(bnaug_data_encoded.columns))
        
        real_data_encoded = real_data_encoded.reindex(columns=all_columns).fillna(0)
        bgan_data_encoded = bgan_data_encoded.reindex(columns=all_columns).fillna(0)
        bnaug_data_encoded = bnaug_data_encoded.reindex(columns=all_columns).fillna(0)

        combined_data = pd.concat([real_data_encoded, bgan_data_encoded, bnaug_data_encoded])
        
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_data)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_scaled)

        n_real = len(real_data)
        n_bgan = len(bgan_data)
        pca_real = pca_result[:n_real]
        pca_bgan = pca_result[n_real:n_real+n_bgan]
        pca_bnaug = pca_result[n_real+n_bgan:]

        plt.figure(figsize=(10,8))
        plt.scatter(pca_real[:,0], pca_real[:,1], alpha=0.5, label='Real', c='blue', s=30)
        plt.scatter(pca_bgan[:,0], pca_bgan[:,1], alpha=0.5, label='BGAN', c='red', s=30)
        plt.scatter(pca_bnaug[:,0], pca_bnaug[:,1], alpha=0.5, label='BN-AUG-SDG', c='green', s=30)
        
        var_explained = pca.explained_variance_ratio_
        plt.xlabel(f'First PC ({var_explained[0]:.1%} variance explained)')
        plt.ylabel(f'Second PC ({var_explained[1]:.1%} variance explained)')
        
        plt.legend()
        plt.title('PCA projection: Real vs BGAN vs BN-AUG-SDG')
        plt.show()

    @staticmethod
    def compute_mmd(X, Y, kernel_bandwidth=None):

        """
        Compute the Maximum Mean Discrepancy (MMD) between two datasets.
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

    @staticmethod
    def calculate_diversity(data):

        """
        Calculate diversity using the Silhouette Score after KMeans clustering.
        """

        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        return score

    @staticmethod
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

    @staticmethod
    def classifier_performance(y_true, y_pred):

        """
        Print precision, recall, F1-score, and AUC for classifier predictions.
        """

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

    def train_and_sample_ctgan(real_train, discrete_columns):

        """
        Train a CTGAN model and sample synthetic data.
        """

        ctgan = CTGAN(epochs=1, discriminator_steps=5, batch_size = 200)
        ctgan.fit(real_train, discrete_columns)
        return ctgan.sample(len(real_train))


    def compute_uncertainty_metrics(scaled_data, feature_names=None, top_n=5, label=""):

        """
        Compute and print mean variance and top uncertain features for synthetic data.
        """

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

        return variances  


    def hyperparameter_search(real_train, real_eval, discrete_columns, bgan_param_grid, bnaug_param_grid, ctgan_param_grid, tvae_param_grid, gc_param_grid, max_runs, n_samples=1000):

        """
        Perform hyperparameter search for multiple SDG models and aggregate metrics.
        """

        results = []

        def run_multiple(method_name, model_class, param_grid):
            for params in ParameterGrid(param_grid):
                metric_runs = []
                for run in range(1, max_runs):  
                    np.random.seed(run)
                    if method_name == "CTGAN":
                        model = model_class(**params)
                        model.fit(real_train, discrete_columns)
                        synthetic = model.sample(n_samples)
                    elif method_name == "TVAE":
                        metadata = SingleTableMetadata()
                        metadata.detect_from_dataframe(data=real_train)
                        model = model_class(metadata, **params)
                        model.fit(real_train)
                        synthetic = model.sample(n_samples)
                    elif method_name == "Gaussian Copula":
                        metadata = SingleTableMetadata()
                        metadata.detect_from_dataframe(data=real_train)
                        model = model_class(metadata)
                        model.fit(real_train)
                        synthetic = model.sample(n_samples)
                    else:
                        model = model_class(**params)
                        model.fit(real_train.sample(frac=1, random_state=run), discrete_columns)
                        synthetic = model.sample(n_samples)

                    metrics = SDGVisualizer.evaluate_sdg_metrics(real_eval, synthetic, model=model)
                    metric_runs.append(metrics)

                    # Convert to DataFrame to check convergence
                    metric_df = pd.DataFrame(metric_runs)
                    converged = True
                    for col in metric_df.columns:
                        mean = metric_df[col].mean()
                        std = metric_df[col].std()
                        se = std / np.sqrt(run)
                        rel_se = se / (abs(mean) + 1e-8)  # Avoid div by 0
                        if rel_se > 0.05:  # 5% relative SE threshold
                            converged = False
                            break
                    if converged and run >= 10:  # Minimum 10 runs to be safe
                        print(f"{method_name} with {params} converged in {run} runs")
                        break

                agg = {col: [metric_df[col].mean(), metric_df[col].std(ddof=0)] for col in metric_df.columns}
                result_dict = {
                    'Method': method_name,
                    **params,
                    **{f"{k}_mean": v[0] for k, v in agg.items()},
                    **{f"{k}_std": v[1] for k, v in agg.items()},
                    'Runs': len(metric_df)
                }
                
                # Add loss config label if both parameters exist
                if 'use_kl_loss' in params and 'use_uncertainty_loss' in params:
                    result_dict['loss_config'] = SDGVisualizer.get_loss_config_label(
                        params['use_kl_loss'], 
                        params['use_uncertainty_loss']
                    )
                
                results.append(result_dict)


        # BGAN grid search
        run_multiple("BGAN", lambda **p: BGAN_SDG(**p).bgan, bgan_param_grid)

        # BN-AUG-SDG grid search
        run_multiple("BN-AUG-SDG", lambda **p: BN_AUG_SDG(**p), bnaug_param_grid)

        # CTGAN grid search
        run_multiple("CTGAN", CTGAN, ctgan_param_grid)

        # TVAE grid search
        run_multiple("TVAE", TVAESynthesizer, tvae_param_grid)

        # Gaussian Copula grid search
        run_multiple("Gaussian Copula", GaussianCopulaSynthesizer, gc_param_grid)

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_sdg_metrics(real_eval, synthetic, model=None):
        """
        Compute MMD, diversity, indistinguishability, uncertainty, and KL divergence metrics.
        """
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

        mmd = SDGVisualizer.compute_mmd(real_eval_scaled_metric, synthetic_scaled_metric)
        diversity = SDGVisualizer.calculate_diversity(synthetic_scaled_metric)
        acc = cross_val_score(
            LogisticRegressionCV(max_iter=1000, cv=3, random_state=42, n_jobs=1),
            np.vstack([real_eval_scaled_metric, synthetic_scaled_metric]),
            np.hstack([np.zeros(len(real_eval_scaled_metric)), np.ones(len(synthetic_scaled_metric))]),
            cv=3, scoring='accuracy', n_jobs=1
        ).mean()

            # --- Uncertainty and KL metrics ---
        uncertainty = None
        kl_div = None
        if model is not None and hasattr(model, "get_last_uncertainty"):
            uncertainty = model.get_last_uncertainty()
        # Fallback: always compute mean variance as uncertainty if not provided
        if uncertainty is None:
            uncertainty = float(np.mean(np.var(synthetic_scaled_metric, axis=0)))
        if model is not None and hasattr(model, "get_last_kl"):
            kl_div = model.get_last_kl()
        # Optionally, fallback for KL if you want:
        if kl_div is None:
            kl_div = float('nan')

        return {
            'MMD': mmd,
            'Diversity': diversity,
            'Indistinguishability': acc,
            'Uncertainty': uncertainty,
            'KL_Divergence': kl_div
        }

    @staticmethod
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
