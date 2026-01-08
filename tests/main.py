






import sys
import os

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from tests.imputation_tests.configurations import Evaluation
from tests.imputation_tests.main import SignificanceTesting, plot_metric
from tests.uncertainty_analysis import UncertaintyAnalysis

# ==========================
# Experiment Configuration
# ==========================
DATASETS = [
    #{"name": "diabetes", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\diabetes_v2", "target": "EarlyReadmission"},
    {"name": "hepatitis", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\mixed_data_hepatisis_dataset", "target": "Category"},
    {"name": "heart", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\baseline_heart_disease_dataset", "target": "diag"},
    #{"name": "diabetes", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\new_datasets\large_diabetes_dataset", "target": "class"},
    {"name": "cancer", "path": r"C:\Users\thoma\Desktop\Publication\Bachelor_Thesis_2025\datasets\Cancer_Dataset.arff", "target": "Class"}
]


# Configuration parameters
N_REPEATS = 1  # Number of times to repeat each experiment
MISSING_RATES = [0.1]  # Keep missing rate reasonable to preserve some target values
RANDOM_SEED = 42  # Base random seed
EPOCHS = 50  # Number of epochs for neural methods
BATCH_SIZE = 100
LEARNING_RATE = 2e-4
EARLY_STOPPING_PATIENCE = 10

# ==========================
# Define Imputation Methods
# ==========================
def create_imputation_methods():
    """Create fresh imputers for each dataset to avoid state contamination."""
    methods = {}

    # Create sklearn-based imputers
    # 1. MissForest (Random Forest based imputer)
    missforest = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
        max_iter=10,  # Increased iterations for better convergence
        initial_strategy='mean',
        min_value=float('-inf'),
        max_value=float('inf')
    )
    missforest.impute_all_missing = lambda X: missforest.fit_transform(X)
    methods['MissForest'] = missforest

    # 2. KNN Imputer
    knn = KNNImputer(n_neighbors=5, weights='uniform')
    knn.impute_all_missing = lambda X: knn.fit_transform(X)
    methods['KNN'] = knn

    # 3. Mean/Mode Imputer
    mean_mode = SimpleImputer(strategy='mean', keep_empty_features=True)
    mean_mode.impute_all_missing = lambda X: mean_mode.fit_transform(X)
    methods['MeanMode'] = mean_mode

    # 4. MICE
    mice = IterativeImputer(
        random_state=RANDOM_SEED,
        sample_posterior=True,
        max_iter=10,
        initial_strategy='mean',
        min_value=float('-inf'),
        max_value=float('inf')
    )
    mice.impute_all_missing = lambda X: mice.fit_transform(X)
    methods['MICE'] = mice

    # Add custom imputers
    methods['BGAIN'] = BGAIN(epochs=EPOCHS)
    methods['BN_AUG_Imputer'] = BN_AUG_Imputer(epochs=EPOCHS)
    
    return methods

# Initialize first set of imputers
imputation_methods = create_imputation_methods()

# ==========================
# Run Experiments
# ==========================
all_quality_results = []
all_impact_results = []

for dataset in DATASETS:
    print(f"\n=== Running experiments on {dataset['name']} ===")
    path = dataset["path"]
    # Flexible ARFF loader that handles STRING attributes and plain ARFF-like files
    def load_arff_flex(fp):
        # read whole file
        with open(fp, 'r', encoding='utf-8') as f:
            txt = f.read()

        # find @DATA section
        parts = txt.split('\n@DATA')
        if len(parts) < 2:
            parts = txt.split('\n@data')
        if len(parts) < 2:
            raise ValueError('No @DATA section found in ARFF file')

        header = parts[0]
        data_section = parts[1]

        # extract attribute names and types from @ATTRIBUTE lines
        cols = []
        categorical_attributes = {}
        for line in header.splitlines():
            line = line.strip()
            if line.upper().startswith('@ATTRIBUTE'):
                # Split preserving quoted strings
                parts = line.split(None, 2)  # max 3 parts: @ATTRIBUTE name type
                if len(parts) >= 3:
                    name = parts[1]
                    type_spec = parts[2]
                    
                    # remove quotes if present
                    if name.startswith('"') and name.endswith('"'):
                        name = name[1:-1]
                    
                    # Check if this is a categorical attribute
                    if type_spec.startswith('{') and type_spec.endswith('}'):
                        # Parse categorical values
                        values = [v.strip().strip('"\'') for v in type_spec[1:-1].split(',')]
                        categorical_attributes[name] = values
                    
                    cols.append(name)

        # parse CSV-like data lines
        data_lines = [l.strip() for l in data_section.splitlines() if l.strip() and not l.strip().startswith('%')]
        rows = []
        for l in data_lines:
            # split by comma but naive (sufficient for typical datasets here)
            vals = [v.strip().strip('"') for v in l.split(',')]
            rows.append(vals)

        df = pd.DataFrame(rows, columns=cols)

        # Handle categorical and numeric columns
        for col in df.columns:
            if col in categorical_attributes:
                # Handle special case for boolean attributes
                if set(v.lower() for v in categorical_attributes[col]) == {'true', 'false'}:
                    # Direct boolean conversion
                    df[col] = df[col].apply(lambda x: str(x).lower() == 'true')
                    print(f"Boolean column {col}: {df[col].value_counts().to_dict()}")
                else:
                    # Map other categorical values consistently
                    df[col] = df[col].astype(str)  # Convert to string first
                    values = categorical_attributes[col]
                    value_map = {}
                    for v in values:
                        value_map[str(v).lower()] = v
                        value_map[str(v).upper()] = v
                        value_map[str(v)] = v  # Exact match
                    
                    # Apply mapping
                    df[col] = df[col].apply(lambda x: value_map.get(str(x).strip(), np.nan))
                    print(f"Categorical column {col}: {df[col].value_counts().to_dict()}")
            else:
                # Try to convert numeric columns
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as e:
                    print(f"Warning: Could not convert {col} to numeric: {e}")

        return df, None

    data, meta = load_arff_flex(path)
    df = pd.DataFrame(data)
    
    # Get target column name first
    target_col = dataset["target"]
    
    # Handle special cases like '?' in the cancer dataset
    print("\nPre-processing data:")
    
    # Special handling for cancer dataset
    if dataset["name"] == "cancer":
        print("Detected cancer dataset - applying specific preprocessing...")
        numeric_columns = ['Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 
                         'Marginal_Adhesion', 'Single_Epi_Cell_Size', 'Bare_Nuclei',
                         'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']
        
        for col in df.columns:
            if col == target_col:
                continue  # Skip target column for now
            
            # Handle known numeric columns
            if any(nc.lower().replace('_', '') == col.lower().replace('_', '') for nc in numeric_columns):
                try:
                    numeric_series = pd.to_numeric(df[col].replace(['?', 'nan', 'null', ''], np.nan))
                    print(f"- Converting known numeric column {col}")
                    df[col] = numeric_series
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not convert known numeric column {col}: {e}")
            else:
                # Handle other columns as potential categorical
                if df[col].dtype == object:
                    df[col] = df[col].replace(['?', 'nan', 'null', ''], np.nan)
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    print(f"- Processing categorical column {col}")
    else:
        # Generic preprocessing for other datasets
        for col in df.columns:
            if col == target_col:
                continue  # Skip target column for now
                
            # Try numeric conversion first
            try:
                numeric_series = pd.to_numeric(df[col].replace(['?', 'nan', 'null', ''], np.nan))
                print(f"- Converting {col} to numeric")
                df[col] = numeric_series
            except (ValueError, TypeError):
                # If numeric conversion fails, handle as categorical
                if df[col].dtype == object:
                    df[col] = df[col].replace(['?', 'nan', 'null', ''], np.nan)
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    print(f"- Processing categorical column {col}")
                    
    print("\nColumn types after preprocessing:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype} (null count: {df[col].isna().sum()})")
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # Extract and process target column
    y = df[target_col].copy()  # Keep a copy for the target
    print(f"\nRaw target column '{target_col}' values:")
    print(f"- Type: {y.dtype}")
    print(f"- Raw value counts: {y.value_counts().to_dict()}")
    
    # Convert target to numeric (handling both binary and multi-class)
    def convert_target(val):
        if isinstance(val, (bool, np.bool_)):
            return 1 if val else 0
        if isinstance(val, (int, float, np.number)):
            return int(val)
        if isinstance(val, str):
            val = val.strip().lower()
            # Binary cases
            if val in ['true', 'yes', '1', 't', 'y', 'malignant', 'abnormal', 'present', 'positive']:
                return 1
            elif val in ['false', 'no', '0', 'f', 'n', 'benign', 'normal', 'absent', 'negative']:
                return 0
                
            # Handle hepatitis dataset specific categories
            if "'0=blood donor'" in val or "'0s=suspect blood donor'" in val:
                return 0
            elif '1=hepatitis' in val:
                return 1
            elif '2=fibrosis' in val:
                return 2
            elif '3=cirrhosis' in val:
                return 3
                
            # Try to extract numeric part from string (e.g., "class 2" -> 2)
            import re
            numbers = re.findall(r'\d+', val)
            if numbers:
                return int(numbers[0])
                
        return np.nan
    
    y = y.apply(convert_target)
    
    # Convert to categorical if more than 2 unique values
    unique_vals = sorted([v for v in y.unique() if not pd.isna(v)])
    if len(unique_vals) > 2:
        print(f"\nDetected multi-class target with values: {unique_vals}")
        # Keep as is for multi-class classification
    else:
        # Convert to binary 0/1 for binary classification
        y = (y > 0).astype(int)
    
    # Verify target conversion and balance
    y_counts = y.value_counts()
    
    # Check if we need to re-encode based on actual dataset
    if len(y_counts) == 1:  # Only one class found - might be encoding issue
        unique_raw = df[target_col].unique()
        print(f"\nWarning: Only one class found after conversion. Raw unique values: {unique_raw}")
        
        # Try dataset-specific encoding if needed
        print(f"Raw target values before specific encoding: {df[target_col].value_counts().to_dict()}")
        
        # Keep a reference to original values
        raw_target = df[target_col].copy()
        
        if any(isinstance(x, str) and ("benign" in x.lower() or "malignant" in x.lower()) for x in raw_target.dropna()):
            print("Detected cancer dataset format - applying specific encoding...")
            y = raw_target.apply(lambda x: 1 if isinstance(x, str) and "malignant" in x.lower() else 0)
        elif any(isinstance(x, str) and x.lower() in ["yes", "no", "positive", "negative"] for x in raw_target.dropna()):
            print("Detected yes/no format - applying specific encoding...")
            y = raw_target.apply(lambda x: 1 if isinstance(x, str) and x.lower() in ["yes", "positive"] else 0)
        elif raw_target.dtype in ['int64', 'float64'] and raw_target.notna().any():
            print("Detected numeric format - normalizing to 0/1...")
            valid_values = raw_target.dropna()
            if not valid_values.empty:
                median = valid_values.median()
                y = (raw_target > median).astype(int)
        
        print(f"Encoded target values: {y.value_counts().to_dict()}")
    
    # Print final processed target statistics
    print(f"\nProcessed target column '{target_col}' statistics:")
    print(f"- Number of values: {len(y)}")
    print(f"- Number of non-null values: {y.notna().sum()}")
    print(f"- Value counts: {y.value_counts().to_dict()}")
    print(f"- Unique values: {sorted(y.unique().tolist())}")
    
    # Final verification
    if y.notna().sum() == 0:
        raise ValueError(f"No valid target values found in column '{target_col}' after conversion")
    elif len(y.unique()) < 2:
        raise ValueError(f"Target column '{target_col}' has only one unique value after conversion")
    
    # Keep target in X but also return it as y for the evaluator
    df[target_col] = y  # Update target column with converted values
    X = df.copy()  # Keep target column in X for missingness patterns
    
    # One-hot encode categorical columns
    print("\nEncoding categorical columns...")
    original_cols = X.columns.tolist()
    X = pd.get_dummies(X)
    new_cols = [c for c in X.columns if c not in original_cols]
    if new_cols:
        print(f"- Added {len(new_cols)} one-hot encoded columns")

    evaluator = Evaluation(imputation_methods, model_type='classification', random_state=RANDOM_SEED)
    multi_run_eval = SignificanceTesting(evaluator, n_repeats=N_REPEATS, random_seed=RANDOM_SEED)

    results = multi_run_eval.run(X, y, MISSING_RATES, dependent_column=target_col, target_column=target_col)

    quality_summary = results['quality_summary']
    impact_summary = results['impact_summary']

    # Add dataset name to results after evaluation
    quality_summary['dataset'] = dataset['name']
    impact_summary['dataset'] = dataset['name']

    all_quality_results.append(quality_summary)
    all_impact_results.append(impact_summary)

    # Optional: run focused uncertainty analysis on this dataset
    try:
        ua = UncertaintyAnalysis(n_imputations=30, random_seed=RANDOM_SEED, debug=True, epochs=EPOCHS, test_size=0.2)
        print(f"Running uncertainty analysis (demo split) on dataset {dataset['name']}")
        # Pass the processed DataFrame X (with target) and run
        ua_summary = ua.analyze(X.drop(columns=[target_col]) if target_col in X.columns else X, missing_rate=0.1)
        print("Uncertainty analysis summary:", ua_summary)
    except Exception:
        # Keep main robust: don't fail the batch if uncertainty analysis fails
        pass

    # ==========================
    # Plot Diagrams
    # ==========================
    try:
        # Set up plotting style - fall back if seaborn not installed as a matplotlib style
        try:
            plt.style.use('seaborn')
        except OSError:
            # fallback to a default available style
            plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'ggplot')
        
        # Print available columns for debugging
        print("\nAvailable columns in quality_summary:")
        print(quality_summary.columns.tolist())
        
        # Quality metrics plots
        # Plot quality metrics
        metrics_to_plot = []
        if 'continuous_rmse' in quality_summary.columns:
            metrics_to_plot.append('continuous_rmse')
        if 'discrete_f1' in quality_summary.columns:
            metrics_to_plot.append('discrete_f1')
            
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            # Create boxplot for each method across patterns/scenarios
            plot_data = quality_summary.copy()
            sns.boxplot(data=plot_data, x='method', y=metric, 
                      hue='pattern', dodge=True)
            
            plt.title(f"{dataset['name']} - {metric.replace('_', ' ').title()}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"results_{dataset['name']}_{metric}.png")
            plt.close()
            
        # Print summary statistics
        print(f"\nResults Summary for {dataset['name']}:")
        print("\nQuality Metrics (Mean ± Std):")
        metrics = ['continuous_rmse', 'discrete_f1']
        for metric in metrics:
            if f"{metric}_mean" in quality_summary.columns:
                means = quality_summary.groupby('method')[f'{metric}_mean'].mean()
                stds = quality_summary.groupby('method')[f'{metric}_std'].mean()
                print(f"\n{metric}:")
                for method in means.index:
                    print(f"  {method}: {means[method]:.3f} ± {stds[method]:.3f}")
                
    except Exception as e:
        print(f"Plotting failed for {dataset['name']}: {e}")
        import traceback
        traceback.print_exc()

# ==========================
# Save Final Results
# ==========================
final_quality = pd.concat(all_quality_results, ignore_index=True)
final_impact = pd.concat(all_impact_results, ignore_index=True)

final_quality.to_csv("results_quality_summary.csv", index=False)
final_impact.to_csv("results_impact_summary.csv", index=False)

print("\n=== Experiment Completed ===")
print("Results saved to results_quality_summary.csv and results_impact_summary.csv")

# ==========================
# Publication-ready plots
# ==========================
try:
    import seaborn as sns
    sns.set(style='whitegrid')
except Exception:
    sns = None

def _save_bar_with_error(df, metric_col, title, filename, ylabel):
    plt.figure(figsize=(10, 6))
    order = df['method'].unique()
    try:
        if sns is not None:
            agg = df.groupby('method')[metric_col].agg(['mean', 'std']).reset_index()
            agg = agg.sort_values('mean')
            ax = sns.barplot(data=agg, x='method', y='mean', yerr=agg['std'].values, palette='tab10')
            ax.set_xlabel('Method')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
            for p, val in zip(ax.patches, agg['mean'].values):
                ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 4), textcoords='offset points')
        else:
            # Fallback simple matplotlib bar with error bars
            agg = df.groupby('method')[metric_col].agg(['mean', 'std']).reset_index()
            agg = agg.sort_values('mean')
            methods = agg['method'].tolist()
            means = agg['mean'].tolist()
            errs = agg['std'].tolist()
            x = range(len(methods))
            plt.bar(x, means, yerr=errs, tick_label=methods)
            plt.xlabel('Method')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Could not save {filename}: {e}")

def _save_boxplot(df, metric_col, title, filename, ylabel):
    plt.figure(figsize=(10, 6))
    try:
        if sns is not None:
            ax = sns.boxplot(data=df, x='method', y=metric_col, palette='tab10')
            ax.set_xlabel('Method')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
        else:
            methods = df['method'].unique()
            data = [df[df['method'] == m][metric_col].dropna().values for m in methods]
            plt.boxplot(data, labels=methods)
            plt.xlabel('Method')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Could not save {filename}: {e}")

# Select metrics available in the final quality DataFrame
if 'continuous_rmse_mean' in final_quality.columns:
    _save_bar_with_error(final_quality, 'continuous_rmse_mean', 'Mean Continuous RMSE by Method', 'pub_continuous_rmse_bar.png', 'Continuous RMSE (mean)')
    _save_boxplot(final_quality, 'continuous_rmse_mean', 'Distribution of Continuous RMSE by Method', 'pub_continuous_rmse_box.png', 'Continuous RMSE')

if 'discrete_f1_mean' in final_quality.columns:
    _save_bar_with_error(final_quality, 'discrete_f1_mean', 'Mean Discrete F1 by Method', 'pub_discrete_f1_bar.png', 'Discrete F1 (mean)')
    _save_boxplot(final_quality, 'discrete_f1_mean', 'Distribution of Discrete F1 by Method', 'pub_discrete_f1_box.png', 'Discrete F1')

print("Publication-ready plots saved: pub_continuous_rmse_bar.png, pub_continuous_rmse_box.png, pub_discrete_f1_bar.png, pub_discrete_f1_box.png (when metrics present)")

