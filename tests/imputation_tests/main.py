from fancyimpute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from .configurations import Evaluation  # Add the import

from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from sklearn.ensemble import RandomForestRegressor
import os
import sys

# Ensure repo root is on sys.path so local packages (bgan, bn_bgan) can be imported
# main.py is in tests/imputation_tests, so go up two levels to reach project root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

class SignificanceTesting:

    """
    Runs multiple repetitions of the evaluation pipeline with different random seeds,
    then aggregates results (mean, std) for imputation quality and downstream impact.
    """

    def __init__(self, base_evaluator: Evaluation, n_repeats=2, random_seed=42):
        """
        Wraps around an Evaluation instance to run multiple repetitions of the evaluation
        with different random seeds, then aggregates results (mean, std).
        
        Args:
            base_evaluator (Evaluation): An instance of the Evaluation class.
            n_repeats (int): Number of repeated runs.
            random_seed (int): Base random seed for reproducibility.
        """
        self.base_evaluator = base_evaluator
        self.n_repeats = n_repeats
        self.random_seed = random_seed

    def run(self, X, y, missing_rates=[0.3, 0.5], **eval_kwargs):
        """
        Run the evaluation pipeline multiple times for each missing rate,
        aggregating the results across runs.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            missing_rates (list of float): List of missing rates to evaluate.
            eval_kwargs: Additional keyword arguments to pass to Evaluation.evaluate_all_conditions.

        Returns:
            dict: Aggregated results with mean and std for both imputation quality and impact on downstream task.
        """
        all_quality_results = []
        all_impact_results = []

        for run_idx in range(self.n_repeats):
            seed = self.random_seed + run_idx
            print(f"\n=== Repetition {run_idx+1}/{self.n_repeats} with seed {seed} ===")

            # we set the random state of the base evaluator to the new seed defined
            self.base_evaluator.random_state = seed

            for missing_rate in missing_rates:
                print(f"\n--- Missing rate: {missing_rate} ---")
                results = self.base_evaluator.evaluate_all_conditions(
                    X, y,
                    missing_rate=missing_rate,
                    **eval_kwargs
                )

                if results is None:
                    print("Skipping this run due to no complete rows after dropping NaNs.")
                    continue  # Skip to next run if no valid results
                
                # Append run and seed info to each result
                for res in results['imputation_quality']:
                    res['missing_rate'] = missing_rate
                    res['run'] = run_idx
                    all_quality_results.append(res)
                for res in results['impact_on_downstream_task']:
                    res['missing_rate'] = missing_rate
                    res['run'] = run_idx
                    all_impact_results.append(res)

        # Convert to DataFrames so we can easily manipulate and summarize
        quality_df = pd.DataFrame(all_quality_results)
        impact_df = pd.DataFrame(all_impact_results)

        # Handle empty results by checking if DataFrames are empty and returning empty summaries
        if quality_df.empty or impact_df.empty:
            print("No results to summarize (all runs may have been skipped due to missing data).")
            return {
                'quality_raw': quality_df,
                'impact_raw': impact_df,
                'quality_summary': pd.DataFrame(),
                'impact_summary': pd.DataFrame()
            }

        # AGGREGATE: mean and std by method, pattern, scenario, missing_rate
        metrics_to_summarize = ['continuous_rmse', 'discrete_f1', 'downstream_f1', 'impact_on_downstream_task']
        
        # First create summary for quality metrics
        quality_metrics = {}
        for metric in ['continuous_rmse', 'discrete_f1', 'downstream_f1']:
            if metric in quality_df.columns:
                grouped = quality_df.groupby(['method', 'pattern', 'scenario', 'missing_rate'])
                mean_vals = grouped[metric].mean().reset_index()
                std_vals = grouped[metric].std().reset_index()
                
                # Rename columns
                mean_vals = mean_vals.rename(columns={metric: f'{metric}_mean'})
                std_vals = std_vals.rename(columns={metric: f'{metric}_std'})
                
                # Merge mean and std
                quality_metrics[metric] = mean_vals.merge(
                    std_vals[['method', 'pattern', 'scenario', 'missing_rate', f'{metric}_std']], 
                    on=['method', 'pattern', 'scenario', 'missing_rate']
                )
        
        # Merge all quality metrics together
        if quality_metrics:
            quality_summary = quality_metrics[list(quality_metrics.keys())[0]]
            for metric in list(quality_metrics.keys())[1:]:
                quality_summary = quality_summary.merge(
                    quality_metrics[metric], 
                    on=['method', 'pattern', 'scenario', 'missing_rate']
                )
        else:
            quality_summary = pd.DataFrame()
            
        # Then create summary for impact metrics
        impact_metrics = {}
        for metric in ['impact_on_downstream_task']:
            if metric in impact_df.columns:
                grouped = impact_df.groupby(['method', 'pattern', 'scenario', 'missing_rate'])
                mean_vals = grouped[metric].mean().reset_index()
                std_vals = grouped[metric].std().reset_index()
                
                # Rename columns
                mean_vals = mean_vals.rename(columns={metric: f'{metric}_mean'})
                std_vals = std_vals.rename(columns={metric: f'{metric}_std'})
                
                # Merge mean and std
                impact_metrics[metric] = mean_vals.merge(
                    std_vals[['method', 'pattern', 'scenario', 'missing_rate', f'{metric}_std']], 
                    on=['method', 'pattern', 'scenario', 'missing_rate']
                )
                    
        # Merge all impact metrics together
        if impact_metrics:
            impact_summary = impact_metrics[list(impact_metrics.keys())[0]]
            for metric in list(impact_metrics.keys())[1:]:
                impact_summary = impact_summary.merge(
                    impact_metrics[metric], 
                    on=['method', 'pattern', 'scenario', 'missing_rate']
                )
        else:
            impact_summary = pd.DataFrame()

        return {
            'quality_raw': quality_df,
            'impact_raw': impact_df,
            'quality_summary': quality_summary,
            'impact_summary': impact_summary
        }

# can be called, not used in the current script
def plot_metric(df, metric, title, ylabel, hue='method'):
        """
        Plots a barplot with error bars showing mean ± std for a given metric.

        Args:
            df: DataFrame with columns like `metric_mean` and `metric_std`
            metric: Base name of the metric, e.g., 'continuous_rmse'
            title: Plot title
            ylabel: Y-axis label
            hue: Column to use for hue (default: 'method')
        """
        plot_df = df.copy()
        
        # Create error bar values
        plot_df['y'] = plot_df[f'{metric}_mean']
        plot_df['yerr'] = plot_df[f'{metric}_std']

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x='pattern', y='y', hue=hue,
            capsize=0.1,
            errorbar=None  
        ) 

        # add manual error bars
        for i, row in plot_df.iterrows():
            plt.errorbar(
                x=i % len(plot_df['pattern'].unique()),  
                y=row['y'],
                yerr=row['yerr'],
                fmt='none',
                capsize=5,
                color='black'
            )

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Missingness Pattern')
        plt.legend(title=hue)
        plt.tight_layout()
        plt.show()

# =========================================================================================================================
# Main execution block to run the evaluation regarding the imputation quality of the model with respct to baseline methods.
# =========================================================================================================================

if __name__ == "__main__":
    """
    Main execution block for benchmarking imputation methods on the Fetal_Dataset.arff dataset.
    This script:
      - Loads and preprocesses the dataset.
      - Defines and configures several imputation methods.
      - Runs repeated evaluation of imputation quality and downstream impact.
      - Aggregates and prints results.
      - Plots summary metrics for comparison.
    """

    # === Experiment Parameters ===
    n_repeats = 5  # Number of repetitions 
    missing_rates = [0.3, 0.5]  # Proportion of missingness to simulate
    random_seed = 42  # For reproducibility
    EPOCHS = 25  # Increased for better convergence

    # === Data Loading and Preprocessing ===
    # Specify dataset paths and their corresponding target columns
    dataset_configs = [
        {
            'path': 'new_datasets/baseline_heart_disease_dataset',
            'target_col': 'diag',
            'discrete_cols': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        },
        {
            'path': 'new_datasets/large_diabetes_dataset',
            'target_col': 'diabetes',
            'discrete_cols': ['gender', 'smoking', 'heart_disease', 'hypertension']
        },
        {
            'path': 'new_datasets/mixed_data_hepatisis_dataset',
            'target_col': 'Category',
            'discrete_cols': ['Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 
                            'SpleenPalpable', 'Spiders', 'Ascites', 'Varices']
        }
    ]
    
    # Select which dataset to use
    dataset = dataset_configs[0]  # Using heart disease dataset
    
    # Load CSV data from the selected dataset directory
    df = pd.read_csv(dataset['path'])
    print("Loaded columns:", df.columns)
    
    target_col = dataset['target_col']
    discrete_columns = dataset['discrete_cols']
    
    # Split features and target
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    # Convert target to numeric if needed
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype == object:
        # Handle categorical target
        y = pd.Categorical(y).codes
    
    # Get dummies only for non-discrete columns that are object type
    non_discrete_cols = [col for col in X.columns if col not in discrete_columns]
    X_dummies = pd.get_dummies(X[non_discrete_cols])
    
    # Combine with discrete columns
    X = pd.concat([X[discrete_columns], X_dummies], axis=1)

    # === Define Imputation Methods ===
    imputation_methods = {
        'RandomForest_MICE': IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=0),
            random_state=0, max_iter=10
        ),
        'MeanMode': SimpleImputer(strategy='mean'),
        'MICE': IterativeImputer(random_state=0, sample_posterior=False, max_iter=10),
        'BGAIN': BGAIN(epochs=EPOCHS),
        'BN_AUG_Imputer': BN_AUG_Imputer(epochs=EPOCHS)
    }
    # Ensure all imputers have a unified interface, for ease of evaluation (for comptability inside the evaluation class)
    for imputer in imputation_methods.values():
        if not hasattr(imputer, "impute_all_missing"):
            imputer.impute_all_missing = imputer.transform

    # Can extend logic to regression tasks too, but datasets would need to be adjusted accordingly, and configurations.py would need to be adjusted slightly too, refer to the class for more details
    # === Evaluation Setup ===
    evaluator = Evaluation(imputation_methods, model_type='classification', discrete_columns=discrete_columns, dataset_name=dataset_configs[0]['name'])

    multi_run_eval = SignificanceTesting(evaluator, n_repeats=n_repeats, random_seed=random_seed)

    # === Run Evaluation ===
    results = multi_run_eval.run(X, y, missing_rates, target_column=target_col)

    # === Output Results ===
    print(results['quality_summary'])
    print(results['impact_summary'])

    quality_summary = results['quality_summary']
    impact_summary = results['impact_summary']
    quality_summary.columns = ['_'.join(col).strip('_') for col in quality_summary.columns.values]
    impact_summary.columns = ['_'.join(col).strip('_') for col in impact_summary.columns.values]
