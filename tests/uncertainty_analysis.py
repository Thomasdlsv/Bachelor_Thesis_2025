import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm

from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from tests.imputation_tests.configurations import mcar, mar, mnar
from sklearn.experimental import enable_iterative_imputer

class UncertaintyAnalysis:
    """
    Comprehensive uncertainty analysis for comparing BGAN and BN-BGAN imputation methods.
    Focuses on uncertainty quantification and calibration metrics.
    """
    
    def __init__(self, n_imputations: int = 30, random_seed: int = 42):
        """
        Initialize the analysis framework.
        
        Args:
            n_imputations: Number of imputations to perform per method
            random_seed: Random seed for reproducibility
        """
        self.n_imputations = n_imputations
        self.random_seed = random_seed
        
        # Initialize imputers with proper discrete column handling
        self.methods = {}
        self.discrete_columns = []  # Will be set when analyzing dataset
        
    def generate_missingness(self, X: pd.DataFrame, pattern: str, rate: float,
                           target_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate missing data according to specified pattern.
        
        Args:
            X: Original complete dataset
            pattern: One of 'MCAR', 'MAR', 'MNAR'
            rate: Missing rate (0-1)
            target_col: Target column to exclude from missingness
            
        Returns:
            Tuple of (corrupted_data, missingness_mask)
        """
        if pattern == 'MCAR':
            X_missing = mcar(X, rate, exclude_cols=[target_col], random_state=self.random_seed)
        elif pattern == 'MAR':
            X_missing = mar(X, rate, exclude_cols=[target_col], random_state=self.random_seed)
        else:  # MNAR
            X_missing = mnar(X, rate, exclude_cols=[target_col], random_state=self.random_seed)
            
        mask = X_missing.isna()
        return X_missing, mask
    
    def _compute_per_cell_stats(self, imputations: List[pd.DataFrame], mask: pd.DataFrame, 
                              X_true: pd.DataFrame) -> Dict:
        """
        Compute per-cell uncertainty statistics and errors.
        
        Args:
            imputations: List of imputed datasets
            mask: Boolean mask indicating missing values
            X_true: Ground truth data
            
        Returns:
            Dictionary containing per-cell statistics
        """
        if not imputations:
            # Handle case where all imputations failed
            nan_array = np.full(mask.shape, np.nan)
            return {
                'means': nan_array[mask],
                'stds': nan_array[mask],
                'errors': nan_array[mask],
                'mean_std': np.nan,
                'median_std': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'coverage_90': np.nan,
                'uncertainty_error_corr': np.nan
            }
            
        try:
            # Convert imputations to numeric arrays and handle any type conversion issues
            numeric_imputations = []
            for imp in imputations:
                try:
                    # Convert to float array, replacing non-numeric values with NaN
                    numeric_imp = imp.astype(float).values
                    numeric_imputations.append(numeric_imp)
                except Exception as e:
                    print(f"Warning: Could not convert imputation to numeric: {str(e)}")
                    continue
            
            if not numeric_imputations:
                raise ValueError("No valid numeric imputations")
                
            # Stack imputations for easy statistics
            stacked = np.stack(numeric_imputations)
            
            # Compute mean and std per cell
            means = np.nanmean(stacked, axis=0)
            stds = np.nanstd(stacked, axis=0)
            
            # Get only masked values
            masked_means = means[mask]
            masked_stds = stds[mask]
            true_values = X_true.astype(float).values[mask]
            
            # Compute errors
            abs_errors = np.abs(masked_means - true_values)
            
            # Compute 90% prediction intervals
            lower = np.nanpercentile(stacked, 5, axis=0)[mask]
            upper = np.nanpercentile(stacked, 95, axis=0)[mask]
            coverage = np.mean((true_values >= lower) & (true_values <= upper))
            
            # Compute correlation between uncertainty and error
            # Handle potential NaN values in correlation
            try:
                uncertainty_error_corr = stats.spearmanr(masked_stds, abs_errors, nan_policy='omit')[0]
                if np.isnan(uncertainty_error_corr):
                    uncertainty_error_corr = 0.0
            except Exception:
                uncertainty_error_corr = 0.0
            
            return {
                'means': masked_means,
                'stds': masked_stds,
                'errors': abs_errors,
                'mean_std': np.nanmean(masked_stds),
                'median_std': np.nanmedian(masked_stds),
                'mae': mean_absolute_error(true_values, masked_means),
                'rmse': np.sqrt(mean_squared_error(true_values, masked_means)),
                'coverage_90': coverage,
                'uncertainty_error_corr': uncertainty_error_corr
            }
        except Exception as e:
            print(f"Error computing statistics: {str(e)}")
            nan_array = np.full(mask.shape, np.nan)
            return {
                'means': nan_array[mask],
                'stds': nan_array[mask],
                'errors': nan_array[mask],
                'mean_std': np.nan,
                'median_std': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'coverage_90': np.nan,
                'uncertainty_error_corr': np.nan
            }
    
    def analyze_dataset(self, X: pd.DataFrame, pattern: str = 'MCAR', 
                       missing_rate: float = 0.2, target_col: str = None) -> Dict:
        """
        Run complete analysis on a dataset.
        
        Args:
            X: Complete dataset
            pattern: Missingness pattern
            missing_rate: Rate of missing values
            target_col: Target column to exclude from missingness
            
        Returns:
            Dictionary containing all analysis results
        """
        np.random.seed(self.random_seed)
        results = {}
        
        # Identify discrete columns (categorical and boolean)
        self.discrete_columns = []
        for col in X.columns:
            if pd.api.types.is_bool_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
                self.discrete_columns.append(col)
            elif X[col].dtype == object:
                # Check if column contains only a small number of unique values
                if len(X[col].unique()) < len(X) * 0.05:  # Less than 5% unique values
                    self.discrete_columns.append(col)
                    
        print(f"Identified discrete columns: {self.discrete_columns}")
        
        # Initialize methods with proper discrete column handling
        self.methods = {
            'BGAN': BGAIN(epochs=1),
            'BN-BGAN': BN_AUG_Imputer(epochs=1)
        }
        
        # Generate missing data
        X_missing, mask = self.generate_missingness(X, pattern, missing_rate, target_col)
        
        print(f"\nAnalyzing {pattern} pattern with {missing_rate*100:.1f}% missing rate")
        print(f"Total missing values: {mask.sum().sum()}")
        
        # Run multiple imputations for each method
        for method_name, method in self.methods.items():
            print(f"\nRunning {method_name}...")
            imputations = []
            
            for i in tqdm(range(self.n_imputations), desc=f"{method_name} imputations"):
                # Reset random seed for each imputation but make it different
                np.random.seed(self.random_seed + i)
                
                try:
                    # Train imputer
                    if hasattr(method, 'fit'):
                        # Check if method accepts discrete_columns parameter
                        if 'discrete_columns' in method.fit.__code__.co_varnames:
                            method.fit(X_missing, discrete_columns=self.discrete_columns)
                        else:
                            method.fit(X_missing)
                    
                    # Impute missing values
                    if hasattr(method, 'impute_all_missing'):
                        if 'discrete_columns' in method.impute_all_missing.__code__.co_varnames:
                            imputed = method.impute_all_missing(X_missing, discrete_columns=self.discrete_columns)
                        else:
                            imputed = method.impute_all_missing(X_missing)
                    
                    # Ensure discrete columns maintain their dtype
                    for col in self.discrete_columns:
                        if pd.api.types.is_bool_dtype(X[col]):
                            imputed[col] = imputed[col].round().astype(bool)
                        elif pd.api.types.is_categorical_dtype(X[col]):
                            imputed[col] = pd.Categorical(imputed[col], categories=X[col].cat.categories)
                            
                    imputations.append(imputed)
                except Exception as e:
                    print(f"Error in imputation {i} with {method_name}: {str(e)}")
                    continue
            
            if imputations:
                # Compute all metrics
                results[method_name] = self._compute_per_cell_stats(imputations, mask, X)
            else:
                print(f"No successful imputations for {method_name}")
            
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Create comprehensive plots of the analysis results.
        
        Args:
            results: Results dictionary from analyze_dataset
            save_path: Path to save plots (optional)
        """
        # Set up the plotting style
        plt.style.use('seaborn')
        
        # 1. Uncertainty Distribution Plot
        plt.figure(figsize=(10, 6))
        data = []
        for method in results:
            data.append(pd.DataFrame({
                'Standard Deviation': results[method]['stds'],
                'Method': method
            }))
        df = pd.concat(data)
        
        sns.boxplot(data=df, x='Method', y='Standard Deviation')
        plt.title('Distribution of Per-Cell Uncertainty')
        if save_path:
            plt.savefig(f"{save_path}_uncertainty_dist.png")
        plt.close()
        
        # 2. Error vs Uncertainty Scatter
        plt.figure(figsize=(12, 5))
        for i, method in enumerate(results, 1):
            plt.subplot(1, 2, i)
            plt.scatter(results[method]['stds'], results[method]['errors'], alpha=0.5)
            plt.xlabel('Standard Deviation (Uncertainty)')
            plt.ylabel('Absolute Error')
            plt.title(f'{method}\nCorr: {results[method]["uncertainty_error_corr"]:.3f}')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_error_vs_uncertainty.png")
        plt.close()
        
        # 3. Summary Metrics Bar Plot
        plt.figure(figsize=(12, 6))
        metrics = ['mean_std', 'mae', 'rmse', 'coverage_90']
        metric_names = ['Mean Uncertainty', 'MAE', 'RMSE', '90% Coverage']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, method in enumerate(results):
            values = [results[method][m] for m in metrics]
            plt.bar(x + i*width, values, width, label=method)
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Summary Metrics Comparison')
        plt.xticks(x + width/2, metric_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_summary_metrics.png")
        plt.close()
        
    def statistical_tests(self, results: Dict) -> Dict:
        """
        Perform statistical tests comparing the methods.
        
        Args:
            results: Results dictionary from analyze_dataset
            
        Returns:
            Dictionary containing test results
        """
        tests = {}
        
        # Get method names
        methods = list(results.keys())
        if len(methods) != 2:
            raise ValueError("Statistical tests are designed for comparing exactly 2 methods")
        
        # 1. Compare uncertainty distributions
        stds_1 = results[methods[0]]['stds']
        stds_2 = results[methods[1]]['stds']
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(stds_1, stds_2)
        
        # Effect size (Cohen's d)
        cohens_d = (np.mean(stds_1) - np.mean(stds_2)) / np.sqrt(
            (np.var(stds_1) + np.var(stds_2)) / 2)
        
        # 2. Compare coverage rates
        n_total = len(stds_1)  # Same as number of missing values
        cover_1 = results[methods[0]]['coverage_90'] * n_total
        cover_2 = results[methods[1]]['coverage_90'] * n_total
        _, coverage_p = stats.proportions_ztest(
            [cover_1, cover_2], 
            [n_total, n_total]
        )
        
        tests['uncertainty_comparison'] = {
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'cohens_d': cohens_d
        }
        
        tests['coverage_comparison'] = {
            'difference': results[methods[0]]['coverage_90'] - results[methods[1]]['coverage_90'],
            'p_value': coverage_p
        }
        
        return tests

if __name__ == "__main__":
    # Dataset Configuration (same as main.py)
    DATASETS = [
        #{"name": "hepatitis", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\mixed_data_hepatisis_dataset", "target": "Category"},
        #{"name": "heart", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\baseline_heart_disease_dataset", "target": "diag"},
        #{"name": "diabetes", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\large_diabetes_dataset", "target": "class"},
        {"name": "cancer", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\datasets\Cancer_Dataset.arff", "target": "Class"}
    ]

    MISSING_RATES = [0.1, 0.2]  # Test different missing rates
    RANDOM_SEED = 42

    # Process each dataset
    for dataset_config in DATASETS:
        print(f"\n=== Running Uncertainty Analysis on {dataset_config['name']} ===")
        
        # Load and preprocess data
        def load_arff_flex(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                txt = f.read()
            
            parts = txt.split('\n@DATA')
            if len(parts) < 2:
                parts = txt.split('\n@data')
            if len(parts) < 2:
                raise ValueError('No @DATA section found in ARFF file')

            header = parts[0]
            data_section = parts[1]

            cols = []
            categorical_attributes = {}
            for line in header.splitlines():
                line = line.strip()
                if line.upper().startswith('@ATTRIBUTE'):
                    parts = line.split(None, 2)
                    if len(parts) >= 3:
                        name = parts[1]
                        type_spec = parts[2]
                        
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
                        value_map = {str(v).lower(): v for v in values}
                        value_map.update({str(v).upper(): v for v in values})
                        value_map.update({str(v): v for v in values})
                        df[col] = df[col].apply(lambda x: value_map.get(str(x).strip(), np.nan))
                else:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to numeric: {e}")
            return df

        df = load_arff_flex(dataset_config["path"])
        
        # Handle special preprocessing for cancer dataset
        if dataset_config["name"] == "cancer":
            print("Applying cancer dataset preprocessing...")
            numeric_columns = ['Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 
                             'Marginal_Adhesion', 'Single_Epi_Cell_Size', 'Bare_Nuclei',
                             'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']
            
            for col in df.columns:
                if col == dataset_config["target"]:
                    continue
                
                if any(nc.lower().replace('_', '') == col.lower().replace('_', '') for nc in numeric_columns):
                    try:
                        df[col] = pd.to_numeric(df[col].replace(['?', 'nan', 'null', ''], np.nan))
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to numeric: {e}")
                else:
                    if df[col].dtype == object:
                        df[col] = df[col].replace(['?', 'nan', 'null', ''], np.nan)
        
        # One-hot encode categorical columns
        df = pd.get_dummies(df)
        
        # Initialize analyzer
        analyzer = UncertaintyAnalysis(n_imputations=30, random_seed=RANDOM_SEED)
        
        for missing_rate in MISSING_RATES:
            print(f"\nAnalyzing with missing rate: {missing_rate}")
            
            # Run analysis for each missingness pattern
            for pattern in ['MCAR', 'MAR', 'MNAR']:
                print(f"\nPattern: {pattern}")
                results = analyzer.analyze_dataset(
                    X=df,
                    pattern=pattern,
                    missing_rate=missing_rate,
                    target_col=dataset_config["target"]
                )
                
                # Generate plots
                save_path = f"uncertainty_{dataset_config['name']}_{pattern}_{missing_rate}"
                analyzer.plot_results(results, save_path=save_path)
                
                # Run statistical tests
                tests = analyzer.statistical_tests(results)
                
                # Print summary
                print("\nAnalysis Summary:")
                for method in results:
                    print(f"\n{method}:")
                    print(f"Mean Uncertainty: {results[method]['mean_std']:.3f}")
                    print(f"MAE: {results[method]['mae']:.3f}")
                    print(f"RMSE: {results[method]['rmse']:.3f}")
                    print(f"90% Coverage: {results[method]['coverage_90']*100:.1f}%")
                    print(f"Uncertainty-Error Correlation: {results[method]['uncertainty_error_corr']:.3f}")
                
                print("\nStatistical Tests:")
                print(f"Uncertainty Difference (Cohen's d): {tests['uncertainty_comparison']['cohens_d']:.3f}")
                print(f"Uncertainty Test p-value: {tests['uncertainty_comparison']['wilcoxon_p']:.3e}")
                print(f"Coverage Difference: {tests['coverage_comparison']['difference']*100:.1f}%")
                print(f"Coverage Test p-value: {tests['coverage_comparison']['p_value']:.3e}")
                
                # Save results to CSV
                results_df = pd.DataFrame({
                    'dataset': dataset_config['name'],
                    'pattern': pattern,
                    'missing_rate': missing_rate,
                    'method': list(results.keys()),
                    'mean_uncertainty': [r['mean_std'] for r in results.values()],
                    'mae': [r['mae'] for r in results.values()],
                    'rmse': [r['rmse'] for r in results.values()],
                    'coverage_90': [r['coverage_90'] for r in results.values()],
                    'uncertainty_error_corr': [r['uncertainty_error_corr'] for r in results.values()]
                })
                
                results_df.to_csv(f"uncertainty_results_{dataset_config['name']}_{pattern}_{missing_rate}.csv", index=False)

    print("\n=== Uncertainty Analysis Completed ===")
    print("Results saved to CSV files and plots generated.")