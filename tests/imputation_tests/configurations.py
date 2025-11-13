import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer


# =========================
# Missingness Pattern Utils
# =========================

def mcar(X, rate, exclude_cols=None, random_state=None):
    """MCAR missingness for DataFrame X at given rate, excluding exclude_cols.
    Ensures at least one value remains observed in each column."""
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    exclude_cols = exclude_cols or []
    cols = list(X_missing.columns)
    n_rows, n_cols = X_missing.shape
    
    # Generate initial mask
    mask = rng.random((n_rows, n_cols)) < rate
    
    # Prevent masking excluded columns
    for c in exclude_cols or []:
        if c in cols:
            j = cols.index(c)
            mask[:, j] = False
            
    # Ensure at least one value remains in each column
    for j in range(n_cols):
        if mask[:, j].all():  # if all True (all masked)
            # Keep at least one random value
            keep_idx = rng.choice(n_rows)
            mask[keep_idx, j] = False
    # Apply NaN values while preserving dtypes
    for j, col in enumerate(cols):
        # Get the column's original dtype
        orig_dtype = X_missing[col].dtype
        
        # Apply NaN values for this column
        col_mask = mask[:, j]
        if col_mask.any():  # Only process if we have any True values
            if np.issubdtype(orig_dtype, np.number):
                # For numeric columns, use np.nan
                X_missing.loc[col_mask, col] = np.nan
            else:
                # For categorical/object columns, use None
                X_missing.loc[col_mask, col] = None
    return X_missing

def mar(X, rate, exclude_cols=None, random_state=None):
    """MAR: missingness in each non-excluded column depends on the previous non-excluded column.
    Ensures at least one value remains observed in each column."""
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    exclude_cols = exclude_cols or []
    cols = [c for c in X.columns if c not in exclude_cols]
    for i, col in enumerate(cols):
        if i == 0:
            continue
        dep = cols[i - 1]
        # select candidates where dep > mean(dep)
        non_na_dep = X[dep].dropna()
        if non_na_dep.empty:
            continue
        threshold = non_na_dep.mean()
        candidate_idx = X.index[X[dep] > threshold].tolist()
        if len(candidate_idx) == 0:
            # If no candidates found based on dependency,
            # keep one random value observed
            all_idx = X.index.tolist()
            if all_idx:
                keep_idx = rng.choice(all_idx)
                candidate_idx = [keep_idx]
            continue
        n_missing = max(1, int(len(candidate_idx) * rate)) if rate > 0 else 0
        chosen = rng.choice(candidate_idx, size=min(len(candidate_idx), n_missing), replace=False)
        
        # Apply NaN values while preserving dtype
        orig_dtype = X_missing[col].dtype
        if np.issubdtype(orig_dtype, np.number):
            X_missing.loc[chosen, col] = np.nan
        else:
            X_missing.loc[chosen, col] = None
    return X_missing

def mnar(X, rate, exclude_cols=None, random_state=None):
    """MNAR: missingness depends on a column's own values (below mean).
    Ensures at least one value remains observed in each column."""
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    exclude_cols = exclude_cols or []
    cols = [c for c in X.columns if c not in exclude_cols]
    for col in cols:
        non_na = X[col].dropna()
        if non_na.empty:
            continue
        threshold = non_na.mean()
        candidate_idx = X.index[X[col] < threshold].tolist()
        if len(candidate_idx) == 0:
            # If no candidates below mean, keep one random value observed
            all_idx = X.index.tolist()
            if all_idx:
                keep_idx = rng.choice(all_idx)
                candidate_idx = [keep_idx]
            continue
        n_missing = int(len(candidate_idx) * rate)
        if n_missing == 0 and len(candidate_idx) > 0:
            n_missing = 1
        chosen = rng.choice(candidate_idx, size=min(len(candidate_idx), n_missing), replace=False)
        
        # Apply NaN values while preserving dtype
        orig_dtype = X_missing[col].dtype
        if pd.api.types.is_bool_dtype(orig_dtype):
            # Convert boolean to float64 before setting NaN
            X_missing[col] = X_missing[col].astype('float64')
            X_missing.loc[chosen, col] = np.nan
        elif np.issubdtype(orig_dtype, np.number):
            X_missing.loc[chosen, col] = np.nan
        else:
            X_missing.loc[chosen, col] = None
    return X_missing

def apply_missingness_pattern(X_train, X_test, pattern='MCAR', scenario='incomplete_train',
                              missing_rate=0.2, target_column=None, random_state=None):
    """
    Apply missingness pattern to data but DO NOT corrupt the target column.
    - pattern: 'MCAR'|'MAR'|'MNAR'
    - scenario: 'train', 'test', or 'both' - specifies which data to corrupt
    - target_column: exclude from masking (REQUIRED)
    - random_state: for reproducibility
    """
    if target_column is None:
        raise ValueError("target_column must be specified to protect target values")
    exclude = [target_column]
    
    # Verify target column exists and has values in relevant dataframes
    if scenario in ['train', 'both'] and not X_train.empty:
        if target_column not in X_train.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        if X_train[target_column].isna().all():
            raise ValueError(f"Target column '{target_column}' contains only NaN values in training data")
            
    if scenario in ['test', 'both'] and not X_test.empty:
        if target_column not in X_test.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data")
        if X_test[target_column].isna().all():
            raise ValueError(f"Target column '{target_column}' contains only NaN values in test data")
    
    X_train_corrupted = X_train.copy()
    X_test_corrupted = X_test.copy()

    if pattern == 'MCAR':
        if scenario in ['train', 'both']:
            X_train_corrupted = mcar(X_train, missing_rate, exclude_cols=exclude, random_state=random_state)
        if scenario in ['test', 'both']:
            X_test_corrupted = mcar(X_test, missing_rate, exclude_cols=exclude, random_state=random_state)

    elif pattern == 'MAR':
        if scenario == 'incomplete_train':
            X_train_corrupted = mar(X_train, missing_rate, exclude_cols=exclude, random_state=random_state)
        else:
            X_test_corrupted = mar(X_test, missing_rate, exclude_cols=exclude, random_state=random_state)

    elif pattern == 'MNAR':
        if scenario == 'incomplete_train':
            X_train_corrupted = mnar(X_train, missing_rate, exclude_cols=exclude, random_state=random_state)
        else:
            X_test_corrupted = mnar(X_test, missing_rate, exclude_cols=exclude, random_state=random_state)

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
        self.imputation_results = {}
        self.discrete_columns = discrete_columns or []
        self.model_type = model_type
        self.random_state = random_state

    def fit_baseline_model(self, X_train, y_train):
        """
        Train baseline model on X_train, y_train (expects no NaNs in X_train).
        Uses a more sensitive model configuration to detect differences.
        """
        # Ensure X_train is DataFrame
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        # instantiate model with parameters suitable for multi-class
        if self.model_type == 'classification':
            # Configure for multi-class with class balancing
            self.baseline_model = RandomForestClassifier(
                n_estimators=50,  # Fewer trees for sensitivity
                max_depth=8,      # Slightly deeper for multi-class
                min_samples_split=2,  # More granular splits
                min_samples_leaf=1,   # Allow specific leaves
                class_weight='balanced_subsample',  # Better for multi-class imbalance
                random_state=self.random_state
            )
        else:
            self.baseline_model = RandomForestRegressor(random_state=self.random_state)

        # Align y
        if not isinstance(y_train, pd.Series):
            try:
                y_train = pd.Series(y_train, index=X_train.index)
            except Exception:
                y_train = pd.Series(y_train)

        # If target column accidentally exists in the feature frame, drop it to avoid leakage
        if isinstance(X_train, pd.DataFrame) and self.target_column in X_train.columns:
            X_train = X_train.drop(columns=[self.target_column])

        common_index = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_index]
        y_train = y_train.loc[common_index]

        # drop rows with missing target
        mask = y_train.notna()
        if not mask.all():
            n_dropped = (~mask).sum()
            print(f"Warning: dropping {n_dropped} training rows because target is NaN.")
            X_train = X_train.loc[mask]
            y_train = y_train.loc[mask]

        if X_train.shape[0] == 0:
            raise ValueError("No training rows left after dropping NaNs from target. Cannot fit baseline model.")

        # Fit the model (X_train must be free of NaNs)
        self.baseline_model.fit(X_train, y_train)

    def evaluate_imputation_quality(self, X_true, y_true, method_name, X_imputed):
        """
        Evaluate imputation quality and downstream prediction on imputed test set.
        """
        # Ensure X_true is DataFrame
        if not isinstance(X_true, pd.DataFrame):
            X_true = pd.DataFrame(X_true)

        # Defensive handling for returned imputed values
        if isinstance(X_imputed, np.ndarray):
            tmp = pd.DataFrame(X_imputed)
            try:
                tmp.index = X_true.index
            except Exception:
                pass
            # reindex to expected columns, fill missing with NaN
            X_imputed = tmp.reindex(columns=list(X_true.columns), fill_value=np.nan)
        elif isinstance(X_imputed, pd.DataFrame):
            # ensure same column order as X_true
            X_imputed = X_imputed.reindex(columns=list(X_true.columns), fill_value=np.nan)
        else:
            # try to coerce
            try:
                X_imputed = pd.DataFrame(X_imputed)
                X_imputed = X_imputed.reindex(columns=list(X_true.columns), fill_value=np.nan)
            except Exception:
                raise ValueError("Unable to coerce X_imputed into DataFrame for evaluation")

        discrete_cols = self.discrete_columns or []
        continuous_cols = [c for c in X_true.columns if c not in discrete_cols]

        results = {'method': method_name}

        # discrete F1
        if discrete_cols:
            f1_scores = []
            for col in discrete_cols:
                mask = X_true[col].notna() & X_imputed[col].notna()
                if mask.sum() == 0:
                    continue
                try:
                    f1 = f1_score(X_true.loc[mask, col], X_imputed.loc[mask, col], average='macro')
                    f1_scores.append(f1)
                except Exception:
                    continue
            results['discrete_f1'] = np.mean(f1_scores) if f1_scores else None

        # continuous RMSE
        if continuous_cols:
            rmses = []
            for col in continuous_cols:
                mask = X_true[col].notna() & X_imputed[col].notna()
                if mask.sum() == 0:
                    continue
                try:
                    rmse = np.sqrt(mean_squared_error(X_true.loc[mask, col], X_imputed.loc[mask, col]))
                    rmses.append(rmse)
                except Exception:
                    continue
            results['continuous_rmse'] = np.mean(rmses) if rmses else None

        # downstream prediction using baseline_model if possible
        try:
            X_pred = X_imputed.copy()
            # remove target column from features if present to avoid leakage
            if isinstance(X_pred, pd.DataFrame) and self.target_column in X_pred.columns:
                X_pred = X_pred.drop(columns=[self.target_column])

            y_pred = self.baseline_model.predict(X_pred)
            if self.model_type == 'classification':
                # Calculate per-class F1 scores for better statistical analysis
                f1_per_class = f1_score(y_true, y_pred, average=None)
                results['downstream_f1'] = np.mean(f1_per_class)
                results['downstream_f1_std'] = np.std(f1_per_class)
            else:
                # For regression, calculate RMSE across all predictions
                residuals = y_true - y_pred
                rmse = np.sqrt(np.mean(residuals ** 2))
                results['downstream_rmse'] = rmse
                results['downstream_rmse_std'] = np.std(residuals)
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            # prediction failed (shape mismatch or classifier not trained) — set downstream metrics to None
            if self.model_type == 'classification':
                results['downstream_f1'] = None
                results['downstream_f1_std'] = None
            else:
                results['downstream_rmse'] = None
                results['downstream_rmse_std'] = None

        return results

    def evaluate_impact_on_downstream_task(self, X_train_imputed, y_train, X_test_full, y_test_full,
                                           X_test_incomplete, y_test_incomplete, X_test_imputed, method_name):
        """
        Compute impact metric relative to baseline trained on complete data.
        The impact measures how much the imputation improves over incomplete data,
        relative to the performance we'd get with complete data.
        
        Impact is calculated as: 100 * (actual_improvement / potential_improvement)
        where:
        - actual_improvement = imputed_score - incomplete_score
        - potential_improvement = baseline_score - incomplete_score
        
        A score of 100 means the imputation fully recovered the baseline performance
        A score of 0 means no improvement over incomplete data
        Negative scores indicate the imputation made things worse
        
        Returns:
            dict: Results including method name and impact score
        """
        # Initialize with NaN impact to handle errors
        results = {
            'method': method_name,
            'impact_on_downstream_task': np.nan,
            'impact_std': np.nan
        }
        # Verify we have missing values to evaluate
        missing_count = X_test_incomplete.isna().sum().sum()
        if missing_count == 0:
            print("    ⚠️ No missing values in test set to evaluate!")
            return {'method': method_name, 'impact_on_downstream_task': np.nan}
            
        # Print detailed missing value distribution
        print("\n    Missing value distribution by column:")
        for col in X_test_incomplete.columns:
            miss_count = X_test_incomplete[col].isna().sum()
            if miss_count > 0:
                print(f"    - {col}: {miss_count} missing ({miss_count/len(X_test_incomplete):.1%})")
                
        # Basic data alignment
        if not isinstance(y_train, pd.Series):
            try:
                y_train = pd.Series(y_train, index=X_train_imputed.index)
            except Exception:
                y_train = pd.Series(y_train)

        # Drop NaN targets and align data
        valid_mask = y_train.notna()
        X_train_final = X_train_imputed.loc[valid_mask]
        y_train_final = y_train.loc[valid_mask]

        # Train baseline model
        try:
            self.fit_baseline_model(X_train_final, y_train_final)
        except Exception as e:
            print(f"    Warning: Could not fit baseline model: {e}")
            return {'method': method_name, 'impact_on_downstream_task': np.nan}

        # Get predictions on full test data (ideal case)
        try:
            # Clean test features for prediction
            X_full_pred = X_test_full.copy()
            if isinstance(X_full_pred, pd.DataFrame):
                # Always drop the target column if present to avoid leakage
                if self.target_column in X_full_pred.columns:
                    X_full_pred = X_full_pred.drop(columns=[self.target_column])
                # Ensure we only use features that were present during training
                X_full_pred = X_full_pred[self.baseline_model.feature_names_in_]
            y_pred_full = self.baseline_model.predict(X_full_pred)
            if self.model_type == 'classification':
                baseline_score = f1_score(y_test_full, y_pred_full, average='macro')
            else:
                baseline_score = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
        except Exception as e:
            print(f"    Warning: Could not evaluate on full test data: {e}")
            return {'method': method_name, 'impact_on_downstream_task': np.nan}

        # Get predictions on incomplete and imputed test data
        try:
            print("\n    Preparing data for evaluation:")
            
            # Create maximally challenging baseline using adversarial filling strategy
            X_test_incomplete_filled = X_test_incomplete.copy()
            for col in X_test_incomplete_filled.columns:
                if not X_test_incomplete_filled[col].isna().any():
                    continue

                col_series = X_test_incomplete_filled[col]
                non_null_vals = col_series.dropna()

                # Detect dummy/one-hot columns (0/1) even if stored as int/float
                is_dummy = False
                try:
                    uniq = set(non_null_vals.unique())
                    if uniq <= {0, 1} or uniq <= {0.0, 1.0}:
                        is_dummy = True
                except Exception:
                    is_dummy = False

                # Boolean columns
                if np.issubdtype(col_series.dtype, np.bool_) or is_dummy:
                    # compute mode (use numeric 0/1 if dummy)
                    if not non_null_vals.empty:
                        mode_val = non_null_vals.mode().iloc[0]
                        try:
                            mode_val = int(mode_val)
                        except Exception:
                            mode_val = 0
                    else:
                        mode_val = 0
                    fill_val = 1 - mode_val  # flip 0/1
                    print(f"    - Filling {col} with {fill_val} (adversarial dummy/boolean)")
                    # fill preserving numeric dtype
                    try:
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna(fill_val)
                    except Exception as e:
                        print(f"    Warning: could not fill dummy {col} with {fill_val}: {e}")
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna(0)

                # Numeric columns (floats/ints)
                elif np.issubdtype(col_series.dtype, np.number):
                    if not non_null_vals.empty:
                        col_mean = non_null_vals.mean()
                        col_std = non_null_vals.std()
                        if not np.isfinite(col_std) or col_std == 0:
                            col_std = max(abs(col_mean) * 0.1, 1.0)
                        col_min = non_null_vals.min()
                        # aggressive adversarial strategy
                        if col_min >= 0 and col_mean > 0:
                            fill_val = -abs(col_mean) * 0.1  # small negative
                        else:
                            fill_val = col_mean - 5 * col_std
                    else:
                        fill_val = -999.0
                    print(f"    - Filling {col} with {fill_val:.3f} (adversarial numeric)")
                    try:
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna(fill_val)
                    except Exception as e:
                        print(f"    Warning: could not fill numeric {col} with {fill_val}: {e}")
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna(0.0)

                # Other (object/categorical) columns
                else:
                    fill_val = "MISSING"
                    print(f"    - Filling {col} with {fill_val} (categorical)")
                    try:
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna(fill_val)
                    except Exception as e:
                        print(f"    Warning: could not fill categorical {col} with {fill_val}: {e}")
                        X_test_incomplete_filled[col] = X_test_incomplete_filled[col].fillna("MISSING")

            # Clean test features for prediction (incomplete)
            X_incomplete_pred = X_test_incomplete_filled.copy()
            if isinstance(X_incomplete_pred, pd.DataFrame):
                if self.target_column in X_incomplete_pred.columns:
                    X_incomplete_pred = X_incomplete_pred.drop(columns=[self.target_column])
                X_incomplete_pred = X_incomplete_pred[self.baseline_model.feature_names_in_]

            # Clean test features for prediction (imputed)
            X_imputed_pred = X_test_imputed.copy()
            if isinstance(X_imputed_pred, pd.DataFrame):
                if self.target_column in X_imputed_pred.columns:
                    X_imputed_pred = X_imputed_pred.drop(columns=[self.target_column])
                X_imputed_pred = X_imputed_pred[self.baseline_model.feature_names_in_]

            y_pred_incomplete = self.baseline_model.predict(X_incomplete_pred)
            y_pred_imputed = self.baseline_model.predict(X_imputed_pred)
            
            if self.model_type == 'classification':
                incomplete_score = f1_score(y_test_incomplete, y_pred_incomplete, average='macro')
                imputed_score = f1_score(y_test_incomplete, y_pred_imputed, average='macro')
            else:
                incomplete_score = np.sqrt(mean_squared_error(y_test_incomplete, y_pred_incomplete))
                imputed_score = np.sqrt(mean_squared_error(y_test_incomplete, y_pred_imputed))

            # Calculate relative improvement vs baseline
            # For classification, higher is better; for regression (RMSE), lower is better
                # Calculate detailed prediction differences and confidence analysis
            print("\n    Detailed Prediction Analysis:")
            
            # Track prediction changes
            y_pred_diffs = (y_pred_imputed != y_pred_incomplete).sum()
            y_pred_diff_rate = y_pred_diffs / len(y_pred_imputed)
            print(f"    - Predictions changed in {y_pred_diffs} cases ({y_pred_diff_rate:.1%} of total)")
            
            if self.model_type == 'classification':
                # For classification: analyze prediction confidence and accuracy
                pred_proba_incomplete = self.baseline_model.predict_proba(X_incomplete_pred)
                pred_proba_imputed = self.baseline_model.predict_proba(X_imputed_pred)
                
                # Calculate confidence differences
                conf_incomplete = np.max(pred_proba_incomplete, axis=1)
                conf_imputed = np.max(pred_proba_imputed, axis=1)
                mean_conf_diff = (conf_imputed - conf_incomplete).mean()
                
                # Calculate accuracy on known values
                correct_incomplete = (y_pred_incomplete == y_test_incomplete).mean()
                correct_imputed = (y_pred_imputed == y_test_incomplete).mean()
                
                print(f"    - Mean confidence change: {mean_conf_diff:.3f}")
                print(f"    - Accuracy: incomplete={correct_incomplete:.3f}, imputed={correct_imputed:.3f}")
                
                # Calculate relative improvement compared to baseline
            if self.model_type == 'classification':
                baseline_score = f1_score(y_test_full, y_pred_full, average='macro')
                incomplete_score = f1_score(y_test_incomplete, y_pred_incomplete, average='macro')
                imputed_score = f1_score(y_test_incomplete, y_pred_imputed, average='macro')
                
                # Calculate F1 improvement relative to baseline
                max_possible_improvement = baseline_score - incomplete_score
                actual_improvement = imputed_score - incomplete_score
                
                if abs(max_possible_improvement) > 1e-10:  # Avoid division by zero
                    impact = 100.0 * (actual_improvement / max_possible_improvement)
                else:
                    impact = 0.0
                    
                print(f"    - F1 scores: baseline={baseline_score:.4f}, incomplete={incomplete_score:.4f}, imputed={imputed_score:.4f}")
                
            else:
                # For regression: Compare RMSE improvements
                baseline_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
                incomplete_rmse = np.sqrt(mean_squared_error(y_test_incomplete, y_pred_incomplete))
                imputed_rmse = np.sqrt(mean_squared_error(y_test_incomplete, y_pred_imputed))
                
                # Calculate RMSE improvement relative to baseline
                max_possible_improvement = incomplete_rmse - baseline_rmse
                actual_improvement = incomplete_rmse - imputed_rmse
                
                if abs(max_possible_improvement) > 1e-10:  # Avoid division by zero
                    impact = 100.0 * (actual_improvement / max_possible_improvement)
                else:
                    impact = 0.0
                    
                print(f"    - RMSE: baseline={baseline_rmse:.4f}, incomplete={incomplete_rmse:.4f}, imputed={imputed_rmse:.4f}")            # Clip impact to reasonable range
            impact = np.clip(impact, -100, 100)
            
            print(f"\n    Impact calculation for {method_name}:")
            print(f"    - Baseline score (full data): {baseline_score:.4f}")
            print(f"    - Incomplete data score: {incomplete_score:.4f}")
            print(f"    - Imputed data score: {imputed_score:.4f}")
            # Store all components for analysis
            results = {
                'method': method_name,
                'impact_on_downstream_task': impact,
                'baseline_score': baseline_score,
                'incomplete_score': incomplete_score,
                'imputed_score': imputed_score,
                'max_improvement': max_possible_improvement,
                'actual_improvement': actual_improvement
            }
            
            print(f"\n    Impact Analysis:")
            print(f"    - Relative improvement: {impact:.1f}%")
            print(f"    - Raw scores: baseline={baseline_score:.4f}, incomplete={incomplete_score:.4f}, imputed={imputed_score:.4f}")
            print(f"    - Improvements: potential={max_possible_improvement:.4f}, actual={actual_improvement:.4f}")
            
            return results
            
        except Exception as e:
            print(f"    Warning: Impact calculation failed: {e}")
            return {
                'method': method_name,
                'impact_on_downstream_task': np.nan,
                'error': str(e)
            }

    def evaluate_all_conditions(self, X, y, missing_rate=0.1, dependent_column=None, target_column=None):
        """
        Evaluate all imputation methods across missingness patterns and training scenarios.
        For each imputer M:
          - fit M on the (possibly incomplete) training data
          - impute training data with M and train baseline model on that imputed training set
          - impute test data with M and evaluate imputation quality and downstream impact
        """
        results = {'imputation_quality': [], 'impact_on_downstream_task': []}

        # normalize common missing markers
        X = X.copy()
        X.replace('?', np.nan, inplace=True)
        X.replace('', np.nan, inplace=True)

        patterns = ['MCAR', 'MAR', 'MNAR']
        scenarios = ['complete_train', 'incomplete_train']

        for pattern in patterns:
            for scenario in scenarios:
                print(f"\nRunning pattern={pattern}, scenario={scenario}")

                # Verify we have enough valid target values before proceeding
                if isinstance(y, pd.Series):
                    valid_targets = y.notna().sum()
                else:
                    valid_targets = np.sum(~pd.isna(y))
                
                if valid_targets < 20:  # Minimum needed for meaningful train/test split
                    print(f"    ⚠️ Too few valid target values ({valid_targets}). Skipping this pattern/scenario.")
                    continue

                # split - stratify if classification to maintain class balance
                try:
                    X_train_full, X_test_full, y_train_full, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=self.random_state,  # Increased test size
                        stratify=y if self.model_type == 'classification' else None
                    )
                except ValueError:  # If stratification fails, proceed without it
                    X_train_full, X_test_full, y_train_full, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=self.random_state  # Increased test size
                    )

                # Store original test data before missingness
                X_test_original = X_test_full.copy()
                
                # Always corrupt test data first
                _, X_test_corrupted = apply_missingness_pattern(
                    X_train_full, X_test_full, pattern=pattern, scenario='test',  # Force test data corruption
                    missing_rate=missing_rate, target_column=target_column,
                    random_state=self.random_state
                )

                print(f"Missing values in test set: {int(X_test_corrupted.isna().sum().sum())}")

                # prepare training set depending on scenario
                if scenario == 'complete_train':
                    # Prefer genuine complete rows if they exist
                    X_train_base = X_train_full.dropna()
                    y_train_base = y_train_full.loc[X_train_base.index]
                    # If none exist, fall back to using the incomplete training set for imputer fitting.
                    # The downstream baseline model will still be trained only on rows that have an observed target.
                    if X_train_base.shape[0] == 0:
                        print("    ⚠️ No fully-complete rows found in training set; falling back to incomplete training set for imputer fitting.")
                        X_train_base = X_train_full.copy()
                        y_train_base = y_train_full.copy()
                    fallback_possible = True
                else:
                    # incomplete_train: we intentionally corrupt the train set (exclude target)
                    X_train_corrupted, _ = apply_missingness_pattern(
                        X_train_full, pd.DataFrame(), pattern=pattern, scenario='train',  # Force train data corruption
                        missing_rate=missing_rate, target_column=target_column, random_state=self.random_state
                    )
                    X_train_base = X_train_corrupted
                    y_train_base = y_train_full
                    fallback_possible = True

                # Function to ensure minimum training coverage
                def ensure_minimal_train_coverage(train_data, test_data, global_data):
                    """Ensures each column has at least one observed value in the training set.
                    
                    Strategy:
                    1. First try to borrow a value from the test set
                    2. If not available, use global statistics
                    3. If still not available, use reasonable defaults based on dtype
                    """
                    result = train_data.copy()
                    for col in train_data.columns:
                        if train_data[col].isna().all():
                            # Try to borrow from test set first
                            if test_data is not None and not test_data[col].isna().all():
                                seed_value = test_data[col].dropna().sample(1, random_state=42).iloc[0]
                            # Otherwise use global statistics
                            elif col in global_data.columns and not global_data[col].isna().all():
                                seed_value = global_data[col].dropna().sample(1, random_state=42).iloc[0]
                            # Last resort: Use type-based defaults
                            else:
                                if np.issubdtype(train_data[col].dtype, np.number):
                                    seed_value = 0
                                else:
                                    seed_value = "unknown"
                            # Insert the seed value in a random position
                            random_idx = np.random.choice(len(result))
                            result.loc[result.index[random_idx], col] = seed_value
                    return result

                # Pre-fill any columns that have no observed values
                fully_nan_cols_train = X_train_base.columns[X_train_base.isna().all()]
                fully_nan_cols_test = X_test_corrupted.columns[X_test_corrupted.isna().all()]
                fully_nan_cols = list(set(fully_nan_cols_train) | set(fully_nan_cols_test))
                
                if fully_nan_cols:
                    print(f"    ⚠️ Ensuring minimum coverage for columns: {fully_nan_cols}")
                    X_train_base = ensure_minimal_train_coverage(X_train_base, X_test_corrupted, X)
                    X_test_corrupted = ensure_minimal_train_coverage(X_test_corrupted, X_train_base, X)

                if fully_nan_cols:
                    print(f"    ⚠️ Pre-filling columns with no observed values: {fully_nan_cols}")
                        # Apply consistent fills to both train and test, but only for fully-NaN columns
                    X_train_base = ensure_minimal_train_coverage(X_train_base, X_test_corrupted, X)
                    X_test_corrupted = ensure_minimal_train_coverage(X_test_corrupted, X_train_base, X)
                    
                    # Verify missingness levels
                    missing_train = X_train_base.isna().sum().sum()
                    total_train_cells = X_train_base.shape[0] * X_train_base.shape[1]
                    missing_test = X_test_corrupted.isna().sum().sum()
                    total_test_cells = X_test_corrupted.shape[0] * X_test_corrupted.shape[1]
                    
                    print(f"\nMissingness statistics:")
                    print(f"- Training set: {missing_train} missing values ({missing_train/total_train_cells:.1%} of cells)")
                    print(f"- Test set: {missing_test} missing values ({missing_test/total_test_cells:.1%} of cells)")
                    
                    if missing_test/total_test_cells < 0.05:
                        print("⚠️ Warning: Very low missingness rate in test set")

                # Drop rows where target is NaN (can't train without labels)
                if not isinstance(y_train_base, pd.Series):
                    try:
                        y_train_base = pd.Series(y_train_base, index=X_train_base.index)
                    except Exception:
                        y_train_base = pd.Series(y_train_base)

                # align indices
                common_idx = X_train_base.index.intersection(y_train_base.index)
                X_train_base = X_train_base.loc[common_idx]
                y_train_base = y_train_base.loc[common_idx]

                # Ensure we have enough valid training data with non-NaN targets
                y_train_non_nan_mask = y_train_base.notna()
                X_train_for_model = X_train_base.loc[y_train_non_nan_mask]
                y_train_for_model = y_train_base.loc[y_train_non_nan_mask]

                if X_train_for_model.shape[0] < 10:  # Minimum threshold for meaningful model training
                    print("    ⚠️ Too few rows with valid target values (<10). Skipping this pattern/scenario combination.")
                    continue

                if X_train_for_model.shape[0] == 0:
                    print("Warning: no rows with valid target to train model, but proceeding with imputation evaluation anyway.")

                # Now evaluate per imputer (each imputer is responsible for imputing training set when needed)
                # Now evaluate per imputer (each imputer is responsible for imputing training set when needed)
                X_train_for_imputers = X_train_base.copy()
                X_test_for_imputers = X_test_corrupted.copy()

                # Convert target to Series for consistent handling
                if not isinstance(y_train_base, pd.Series):
                    try:
                        y_train_base = pd.Series(y_train_base, index=X_train_base.index)
                    except Exception:
                        y_train_base = pd.Series(y_train_base)

                # Ensure we have enough valid target values for model training
                valid_target_mask = y_train_base.notna()
                valid_targets = valid_target_mask.sum()
                
                if valid_targets < 10:  # Minimum threshold for model training
                    print(f"    ⚠️ Too few valid target values ({valid_targets}). Skipping this pattern/scenario.")
                    continue

                from tests.main import create_imputation_methods
                fresh_imputers = create_imputation_methods()
                
                for method_name, imputer in fresh_imputers.items():
                    print(f"  Evaluating imputer: {method_name}")

                    try:
                        # For sklearn imputers, we use fit_transform directly via lambda
                        if isinstance(imputer, (SimpleImputer, IterativeImputer)):
                            X_train_imputed = imputer.impute_all_missing(X_train_for_imputers)
                            X_test_imputed = imputer.transform(X_test_for_imputers)  # use fitted state
                        else:
                            # For BGAIN and BN_AUG_Imputer, we follow their API
                            if hasattr(imputer, "fit"):
                                imputer.fit(X_train_for_imputers)
                            X_train_imputed = imputer.impute_all_missing(X_train_for_imputers)
                            X_test_imputed = imputer.impute_all_missing(X_test_for_imputers)
                    except Exception as e:
                        print(f"    Imputer {method_name} failed: {str(e)}. Skipping this imputer.")
                        continue

                    # Use pre-filled data for imputer training
                    X_train_for_imputers = X_train_base.copy()
                    X_test_for_imputers = X_test_corrupted.copy()

                    # Remove any remaining fully NaN rows (should be rare now)
                    remaining_fully_nan_rows = X_train_for_imputers.index[X_train_for_imputers.isna().all(axis=1)]
                    if len(remaining_fully_nan_rows) > 0:
                        print(f"    ⚠️ Dropping {len(remaining_fully_nan_rows)} rows that remain fully NaN.")
                        X_train_for_imputers = X_train_for_imputers.drop(index=remaining_fully_nan_rows)
                        y_train_for_model = y_train_for_model.drop(index=remaining_fully_nan_rows, errors="ignore")


                    # ✅ 2. Detect and drop fully NaN rows (rare but can break imputers)
                    fully_nan_rows = X_train_for_imputers.index[X_train_for_imputers.isna().all(axis=1)]
                    if len(fully_nan_rows) > 0:
                        print(f"    ⚠️ Dropping {len(fully_nan_rows)} rows that are fully NaN in train.")
                        X_train_for_imputers = X_train_for_imputers.drop(index=fully_nan_rows)
                        y_train_for_model = y_train_for_model.drop(index=fully_nan_rows, errors="ignore")

                    # Check viability after placeholders
                    if X_train_for_imputers.shape[0] == 0:
                        print(f"    ⚠️ No rows left after placeholder handling. Skipping {method_name}.")
                        continue

                    # ✅ 3. Fit imputer safely
                    try:
                        if hasattr(imputer, "fit"):
                            if "discrete_columns" in imputer.fit.__code__.co_varnames:
                                imputer.fit(X_train_for_imputers, discrete_columns=self.discrete_columns or [])
                            else:
                                imputer.fit(X_train_for_imputers)
                    except Exception as e:
                        print(f"    Imputer {method_name} fit failed: {e}. Skipping this imputer.")
                        continue

                    # ✅ 4. Impute train and test
                    try:
                        X_train_imputed = imputer.impute_all_missing(X_train_for_imputers)
                        X_test_imputed = imputer.impute_all_missing(X_test_for_imputers)
                    except Exception as e:
                        print(f"    Imputer {method_name} imputation failed: {e}. Skipping this imputer.")
                        continue

                    # Ensure imputed outputs are DataFrames with correct index/columns
                    if isinstance(X_train_imputed, np.ndarray):
                        try:
                            X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train_for_imputers.columns, index=X_train_for_imputers.index)
                        except Exception:
                            X_train_imputed = pd.DataFrame(X_train_imputed, index=X_train_for_imputers.index)
                    elif isinstance(X_train_imputed, pd.DataFrame):
                        X_train_imputed = X_train_imputed.reindex(columns=X_train_for_imputers.columns, fill_value=np.nan)

                    if isinstance(X_test_imputed, np.ndarray):
                        try:
                            X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test_for_imputers.columns, index=X_test_for_imputers.index)
                        except Exception:
                            X_test_imputed = pd.DataFrame(X_test_imputed, index=X_test_for_imputers.index)
                    elif isinstance(X_test_imputed, pd.DataFrame):
                        X_test_imputed = X_test_imputed.reindex(columns=X_test_for_imputers.columns, fill_value=np.nan)

                    # ✅ 5. Evaluate using new metrics
                    from tests.imputation_tests.evaluation_metrics import evaluate_imputer
                    
                    # Record which values were actually masked
                    missing_mask = X_test_for_imputers.isna()
                    
                    def model_builder():
                        """Create a fresh model instance."""
                        if self.model_type == 'classification':
                            return RandomForestClassifier(
                                n_estimators=50,
                                max_depth=8,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                class_weight='balanced_subsample',
                                random_state=self.random_state
                            )
                        else:
                            return RandomForestRegressor(random_state=self.random_state)
                    
                    # Evaluate imputer with comprehensive metrics
                    result = evaluate_imputer(
                        imputer=imputer,
                        X_train=X_train_for_imputers, 
                        X_test=X_test_for_imputers,
                        y_train=y_train_for_model,
                        y_test=y_test,
                        X_original=X_test_original,  # Original data before missingness
                        mask_test=missing_mask,  # Which values were masked
                        scenario=scenario,
                        model_builder=model_builder,
                        method_name=method_name,
                        pattern=pattern,
                        missing_rate=missing_rate,
                        dataset=self.dataset_name if hasattr(self, 'dataset_name') else 'unknown'
                    )
                    
                    # Split into quality and impact results
                    quality_result = {
                        'method': method_name,
                        'pattern': pattern,
                        'scenario': scenario,
                        'continuous_rmse': result['continuous_rmse_mean'],
                        'categorical_acc': result['categorical_acc_mean']
                    }
                    results['imputation_quality'].append(quality_result)
                    
                    impact_result = {
                        'method': method_name,
                        'pattern': pattern,
                        'scenario': scenario,
                        'impact_on_downstream_task': result['impact_on_downstream_task_mean'],
                        'downstream_f1': result['downstream_f1_mean']
                    }
                    results['impact_on_downstream_task'].append(impact_result)


        return results
