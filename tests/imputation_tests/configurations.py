import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bgan.utility.bgan_imp import BGAIN
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import IterativeImputer  # MICE
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer


# =========================
# Missingness Pattern Utils
# =========================

def mcar(X, rate):
    """Apply MCAR missingness to DataFrame X at the given rate."""
    X_missing = X.copy()
    mask = np.random.rand(*X.shape) < rate
    X_missing[mask] = np.nan
    return X_missing

def mar(X, rate):
    """Apply MAR missingness: each column depends on the previous column."""
    X_missing = X.copy()
    for i, col in enumerate(X.columns):
        if i == 0:  # skip first column
            continue
        dependent_col = X.columns[i - 1]
        threshold = X[dependent_col].mean()
        mask = X[dependent_col] > threshold
        missing_indices = X[mask].sample(frac=rate, random_state=42).index
        X_missing.loc[missing_indices, col] = np.nan
    return X_missing

def mnar(X, rate):
    """Apply MNAR missingness: missingness depends on values below the mean."""
    X_missing = X.copy()
    for col in X.columns:
        threshold = X[col].mean()
        mask = X[col] < threshold
        n_candidates = mask.sum()
        n_missing = int(rate * n_candidates)
        if n_missing == 0 and n_candidates > 0:
            n_missing = 1
        if n_candidates > 0:
            missing_indices = X[mask].sample(n=n_missing, random_state=42).index
            X_missing.loc[missing_indices, col] = np.nan
    return X_missing


def apply_missingness_pattern(X_train, X_test, pattern='MCAR', scenario='incomplete_train', missing_rate=0.2):
    """
    Apply a missingness pattern to train and/or test data.
    """
    X_train_corrupted = X_train.copy()
    X_test_corrupted = X_test.copy()

    if pattern == 'MCAR':
        if scenario == 'incomplete_train':
            X_train_corrupted = mcar(X_train, missing_rate)
        else:  # complete_train
            X_test_corrupted = mcar(X_test, missing_rate)

    elif pattern == 'MAR':
        if scenario == 'incomplete_train':
            X_train_corrupted = mar(X_train, missing_rate)
        else:  # complete_train
            X_test_corrupted = mar(X_test, missing_rate)

    elif pattern == 'MNAR':
        if scenario == 'incomplete_train':
            X_train_corrupted = mnar(X_train, missing_rate)
        else:  # complete_train
            X_test_corrupted = mnar(X_test, missing_rate)

    return X_train_corrupted, X_test_corrupted


# ======================
# Evaluation Pipeline
# ======================

class Evaluation:
    """
    Class for evaluating imputation methods on imputation quality and downstream task impact.
    """
    def __init__(self, imputation_methods, model_type='classification', random_state=42, discrete_columns=None):
        self.imputation_methods = imputation_methods
        self.model_type = model_type
        self.random_state = random_state
        self.baseline_model = None
        self.imputation_results = {}
        self.discrete_columns = discrete_columns

    def fit_baseline_model(self, X_train, y_train, use_imputed_train=False, imputer=None):
        """
        Train baseline model.
        If use_imputed_train=True, impute missing values in X_train before training.
        """
        if use_imputed_train and imputer is not None:
            X_train = imputer.impute_all_missing(X_train)

        if self.model_type == 'classification':
            self.baseline_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        else:
            self.baseline_model = RandomForestRegressor(random_state=self.random_state)

        self.baseline_model.fit(X_train, y_train)


    def evaluate_imputation_quality(self, X_true, y_true, method_name, X_imputed):
        """
        Evaluate imputation quality separately for discrete and continuous features.
        Discrete: F1-score per column (macro average over all discrete cols)
        Continuous: RMSE averaged over continuous cols
        
        Also evaluates downstream task using self.baseline_model on imputed data.
        """
        if isinstance(X_imputed, np.ndarray):
            X_imputed = pd.DataFrame(X_imputed, columns=X_true.columns, index=X_true.index)
        discrete_cols = self.discrete_columns or []
        continuous_cols = [col for col in X_true.columns if col not in discrete_cols]
        
        results = {'method': method_name}
        
        # Evaluate discrete columns with F1
        if discrete_cols:
            f1_scores = []
            for col in discrete_cols:
                mask = X_true[col].notna()
                if mask.sum() == 0:
                    continue
                f1 = f1_score(X_true.loc[mask, col], X_imputed.loc[mask, col], average='macro')
                f1_scores.append(f1)
            if f1_scores:
                results['discrete_f1'] = np.mean(f1_scores)
            else:
                results['discrete_f1'] = None

        # Evaluate continuous columns with RMSE
        if continuous_cols:
            rmses = []
            for col in continuous_cols:
                mask = X_true[col].notna()
                if mask.sum() == 0:
                    continue
                rmse = np.sqrt(mean_squared_error(X_true.loc[mask, col], X_imputed.loc[mask, col]))
                rmses.append(rmse)
            if rmses:
                results['continuous_rmse'] = np.mean(rmses)
            else:
                results['continuous_rmse'] = None

        # Evaluate downstream task metric (classification/regression) on imputed data
        y_pred = self.baseline_model.predict(X_imputed)
        if self.model_type == 'classification':
            f1 = f1_score(y_true, y_pred, average='macro')
            results['downstream_f1'] = f1
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            results['downstream_rmse'] = rmse

        return results

    def evaluate_impact_on_downstream_task(self, X_train_full, y_train_full, X_test_full, y_test_full,
                                       X_test_incomplete, y_test_incomplete, imputed_X, method_name):
        """
        Evaluate the impact of imputation on downstream task performance.
        """

        # 1. Baseline: model on full train, full test
        self.fit_baseline_model(X_train_full, y_train_full)
        baseline_result = self.evaluate_imputation_quality(X_test_full, y_test_full, method_name, X_test_full)
        incomplete_result = self.evaluate_imputation_quality(X_test_incomplete, y_test_incomplete, method_name, X_test_incomplete)
        imputed_result = self.evaluate_imputation_quality(X_test_incomplete, y_test_incomplete, method_name, imputed_X)

        # Choose the correct metric for downstream task
        if self.model_type == 'classification':
            metric = 'downstream_f1'
        else:
            metric = 'downstream_rmse'

        baseline_score = baseline_result.get(metric)
        incomplete_score = incomplete_result.get(metric)
        imputed_score = imputed_result.get(metric)

        # Avoid division by zero
        if baseline_score is None or baseline_score == 0:
            impact = np.nan
        else:
            impact = (imputed_score - incomplete_score) / baseline_score * 100

        return {'method': method_name, 'impact_on_downstream_task': impact}

    def evaluate_all_conditions(self, X, y, missing_rate=0.1, dependent_column='education-num', target_column='Class'):
        """
        Evaluate all imputation methods across all missingness patterns and training scenarios.
        Properly simulates MNAR by introducing missingness into both train and test sets.
        """
        results = {
            'imputation_quality': [],
            'impact_on_downstream_task': []
        }

        missingness_patterns = ['MCAR', 'MAR', 'MNAR']
        scenarios = ['complete_train', 'incomplete_train']

        for pattern in missingness_patterns:
            for scenario in scenarios:
                print(f"\nRunning pattern={pattern}, scenario={scenario}")

                X_train_full, X_test_full, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

                # Always apply missingness to test set, using scenario='complete_train' to simulate realistic test-time missingness
                _, X_test_corrupted = apply_missingness_pattern(
                    X_train_full, X_test_full,
                    pattern=pattern,
                    scenario='complete_train',  # Always simulate missingness at test time
                    missing_rate=missing_rate
                )

                print(f"Missing values in test set: {X_test_corrupted.isna().sum().sum()}")  # Debug line


                # Apply missingness to training set
                if scenario == 'complete_train':
                    X_train = X_train_full.dropna()
                    y_train = y_train_full[X_train.index]
                elif scenario == 'incomplete_train':
                    X_train, _ = apply_missingness_pattern(
                        X_train_full, X_test_full,
                        pattern=pattern,
                        scenario=scenario,
                        missing_rate=missing_rate
                    )
                    y_train = y_train_full

                # Drop rows with NaNs before fitting model
                X_train_clean = X_train.dropna()
                y_train_clean = y_train[X_train_clean.index]

                if X_train_clean.shape[0] == 0:
                    print("Warning: No complete rows in X_train after dropping NaNs. Skipping this run.")
                    return None  # or handle as appropriate for your pipeline
                self.fit_baseline_model(X_train_clean, y_train_clean)

                if len(X_train_clean) == 0:
                    print(f"Skipping {pattern}, {scenario}: No complete rows in training.")
                    continue

                self.fit_baseline_model(X_train_clean, y_train_clean)

                for method_name, imputer in self.imputation_methods.items():
                    print(f"  Evaluating imputer: {method_name}")

                    X_train_imp = X_train.dropna()
                    y_train_imp = y_train[X_train_imp.index]

                    # Check for sufficient samples in each column
                    min_samples_per_col = X_train_imp.count().min()
                    if min_samples_per_col < 2:
                        print(f"Skipping {pattern}, {scenario}, {method_name}: Not enough non-missing samples in at least one column (min={min_samples_per_col})")
                        continue

                    # Define proper training data for imputation
                    if scenario == 'complete_train':
                        X_train_for_imputation = X_train  # Already complete
                        y_train_for_model = y_train

                    elif scenario == 'incomplete_train':
                        X_train_for_imputation = X_train  # Contains missing values
                        y_train_for_model = y_train

                    # Detect continuous vs discrete columns
                    continuous_cols = [col for col in X_train_for_imputation.columns if col not in (self.discrete_columns or [])]

                    # Special handling for BGAIN
                    if method_name == 'BGAIN':
                        # Drop rows where any continuous column has a NaN
                        X_train_filtered = X_train_for_imputation.dropna(subset=continuous_cols)
                        y_train_filtered = y_train_for_model[X_train_filtered.index]

                        if hasattr(imputer, "fit") and "discrete_columns" in imputer.fit.__code__.co_varnames:
                            imputer.fit(X_train_filtered, discrete_columns=self.discrete_columns or [])
                        else:
                            imputer.fit(X_train_filtered)

                    else:
                        if hasattr(imputer, "fit") and "discrete_columns" in imputer.fit.__code__.co_varnames:
                            imputer.fit(X_train_for_imputation, discrete_columns=self.discrete_columns or [])
                        else:
                            imputer.fit(X_train_for_imputation)


                    # Impute both training and test data
                    X_train_imputed = imputer.impute_all_missing(X_train_for_imputation)
                    X_test_imputed = imputer.impute_all_missing(X_test_corrupted)

                    # Only evaluate test rows where imputation happened
                    missing_rows_mask = X_test_corrupted.isna().any(axis=1)
                    if missing_rows_mask.sum() == 0:
                        print("  No missing values in test set to evaluate.")
                        continue

                    y_test_subset = y_test[missing_rows_mask]
                    X_test_subset = X_test_corrupted[missing_rows_mask]
                    X_test_imputed_subset = X_test_imputed[missing_rows_mask]

                    # Evaluate imputation quality
                    quality_result = self.evaluate_imputation_quality(
                        X_test_subset, y_test_subset, method_name, X_test_imputed_subset
                    )
                    quality_result.update({'pattern': pattern, 'scenario': scenario})
                    results['imputation_quality'].append(quality_result)

                    # Evaluate downstream impact using imputed train + test
                    impact_result = self.evaluate_impact_on_downstream_task(
                        X_train_for_imputation, y_train_for_model,
                        X_test_subset, y_test_subset,
                        X_test_subset, y_test_subset,
                        X_test_imputed_subset, method_name,
                    )
                    impact_result.update({'pattern': pattern, 'scenario': scenario})
                    results['impact_on_downstream_task'].append(impact_result)
                   
        return results

