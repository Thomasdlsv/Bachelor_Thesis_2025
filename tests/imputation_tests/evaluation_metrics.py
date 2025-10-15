"""Evaluation metrics for imputation quality and downstream impact."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
from copy import deepcopy

def compute_numeric_rmse(X_original: pd.DataFrame, X_imputed: pd.DataFrame, mask: pd.DataFrame):
    """Compute RMSE on numeric columns at positions that were originally missing."""
    numeric_cols = X_original.select_dtypes(include=[np.number]).columns
    mse_total, count = 0.0, 0

    # Ensure both are DataFrames with matching structure
    if isinstance(X_imputed, np.ndarray):
        X_imputed = pd.DataFrame(X_imputed, columns=X_original.columns, index=X_original.index)
    else:
        X_imputed = X_imputed.reindex(columns=X_original.columns, index=X_original.index)

    # Process numeric columns
    for col in numeric_cols:
        missing_idx = mask[col]
        if missing_idx.any():
            try:
                # Extract values and ensure they're numeric
                y_true = pd.to_numeric(X_original.loc[missing_idx, col], errors='coerce')
                y_pred = pd.to_numeric(X_imputed.loc[missing_idx, col], errors='coerce')
                
                # Only compare where both values are valid numbers
                valid_mask = y_true.notna() & y_pred.notna()
                if valid_mask.any():
                    mse = mean_squared_error(
                        y_true[valid_mask],
                        y_pred[valid_mask]
                    )
                    mse_total += mse
                    count += 1
            except Exception as e:
                print(f"    Warning: RMSE calculation failed for column {col}: {e}")
                continue

    return np.sqrt(mse_total / max(count, 1)) if count > 0 else 0.0


def compute_categorical_accuracy(X_original: pd.DataFrame, X_imputed: pd.DataFrame, mask: pd.DataFrame):
    """Compute accuracy for categorical columns at positions that were originally missing."""
    cat_cols = X_original.select_dtypes(exclude=[np.number]).columns
    correct, total = 0, 0

    # Ensure both are DataFrames with matching structure
    if isinstance(X_imputed, np.ndarray):
        X_imputed = pd.DataFrame(X_imputed, columns=X_original.columns, index=X_original.index)
    else:
        X_imputed = X_imputed.reindex(columns=X_original.columns, index=X_original.index)

    for col in cat_cols:
        missing_idx = mask[col]
        if missing_idx.any():
            try:
                true_vals = X_original.loc[missing_idx, col]
                pred_vals = X_imputed.loc[missing_idx, col]
                
                # Convert to string for consistent comparison
                true_vals = true_vals.astype(str)
                pred_vals = pred_vals.astype(str)
                
                # Only compare where both values are non-null
                valid_mask = true_vals.notna() & pred_vals.notna()
                if valid_mask.any():
                    correct += (true_vals[valid_mask] == pred_vals[valid_mask]).sum()
                    total += valid_mask.sum()
            except Exception as e:
                print(f"    Warning: Categorical accuracy calculation failed for column {col}: {e}")
                continue

    return correct / total if total > 0 else np.nan


def evaluate_imputer(imputer, X_train, X_test, y_train, y_test,
                     X_original, mask_test,
                     scenario, model_builder,
                     method_name, pattern, missing_rate, dataset):
    """
    Evaluate imputer both on reconstruction quality (RMSE / categorical accuracy)
    and downstream model impact (F1).
    """
    # ✅ Step 1: Impute missing values
    if hasattr(imputer, "fit"):
        imputer.fit(X_train)
        X_train_imp = imputer.impute_all_missing(X_train)
        X_test_imp = imputer.impute_all_missing(X_test)
    else:
        X_train_imp = imputer.impute_all_missing(X_train)
        X_test_imp = imputer.impute_all_missing(X_test)

    # ✅ Step 2: Convert numpy arrays to DataFrames if needed
    if isinstance(X_train_imp, np.ndarray):
        X_train_imp = pd.DataFrame(X_train_imp, columns=X_train.columns, index=X_train.index)
    if isinstance(X_test_imp, np.ndarray):
        X_test_imp = pd.DataFrame(X_test_imp, columns=X_test.columns, index=X_test.index)

    # ✅ Step 3: Match dtypes to original (important to avoid FutureWarnings)
    for col in X_original.columns:
        try:
            X_train_imp[col] = X_train_imp[col].astype(X_original[col].dtype, errors="ignore")
            X_test_imp[col] = X_test_imp[col].astype(X_original[col].dtype, errors="ignore")
        except Exception:
            pass

    # ✅ Step 4: Compute RMSE and Categorical Accuracy
    rmse = compute_numeric_rmse(X_original, X_test_imp, mask_test)
    cat_acc = compute_categorical_accuracy(X_original, X_test_imp, mask_test)

    # ✅ Step 4: Downstream performance — train baseline model on clean data
    clf_baseline = model_builder()
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    f1_baseline = f1_score(y_test, y_pred_baseline, average="macro")

    # ✅ Step 5: Downstream performance after imputation
    clf_imp = model_builder()
    clf_imp.fit(X_train_imp, y_train)
    y_pred_imp = clf_imp.predict(X_test_imp)
    f1_imp = f1_score(y_test, y_pred_imp, average="macro")

    # ✅ Step 6: Compute relative impact
    impact = f1_imp - f1_baseline

    # ✅ Step 7: Package results
    result = {
        "method": method_name,
        "pattern": pattern,
        "scenario": scenario,
        "missing_rate": missing_rate,
        "continuous_rmse_mean": rmse,
        "categorical_acc_mean": cat_acc,
        "downstream_f1_mean": f1_imp,
        "impact_on_downstream_task_mean": impact,
        "dataset": dataset
    }

    return result