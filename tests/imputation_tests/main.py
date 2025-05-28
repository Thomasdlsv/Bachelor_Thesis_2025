from fancyimpute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer

from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from tests.imputation_tests.configurations import Evaluation
from sklearn.ensemble import RandomForestRegressor

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
        quality_summary = quality_df.drop(columns=['run']).groupby(
            ['method', 'pattern', 'scenario', 'missing_rate']
        ).agg(['mean', 'std']).reset_index()

        impact_summary = impact_df.drop(columns=['run']).groupby(
            ['method', 'pattern', 'scenario', 'missing_rate']
        ).agg(['mean', 'std']).reset_index()

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
    n_repeats = 2 # Number of repetitions 
    missing_rates = [0.1] # Proportion of missingness to simulate
    random_seed = 42 # For reproducibility
    EPOCHS = 1

    # === Data Loading and Preprocessing ===
    # Load ARFF data
    # Can change the dataset to any dataset (refer to datasets folder)
    data, meta = arff.loadarff('datasets/Fetal_Dataset.arff')
    df = pd.DataFrame(data)

    print("Loaded columns:", df.columns)
    
    # Convert byte columns to string (if needed)
    for col in df.select_dtypes([object]).columns:
        try:
            df[col] = df[col].str.decode('utf-8')
        except AttributeError:
            pass  # this means that it's already string

    target_col = 'Class'
    X = df.drop(columns=target_col)
    y = df[target_col]

    X = pd.get_dummies(X)
    discrete_columns = []

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
    evaluator = Evaluation(imputation_methods, model_type='classification', discrete_columns=discrete_columns)

    multi_run_eval = SignificanceTesting(evaluator, n_repeats=n_repeats, random_seed=random_seed)

    # === Run Evaluation ===
    # You can adjust dependent_column/target_column as needed for your dataset
    results = multi_run_eval.run(X, y, missing_rates, dependent_column='V1', target_column='V1')

    # === Output Results ===
    print(results['quality_summary'])
    print(results['impact_summary'])

    quality_summary = results['quality_summary']
    impact_summary = results['impact_summary']
    quality_summary.columns = ['_'.join(col).strip('_') for col in quality_summary.columns.values]
    impact_summary.columns = ['_'.join(col).strip('_') for col in impact_summary.columns.values]
