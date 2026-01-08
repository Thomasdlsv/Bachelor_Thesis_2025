




"""Focused uncertainty ablation: compare BGAIN vs BN_AUG_Imputer.

This module exposes UncertaintyAnalysis which runs a small number of
imputations per method, computes per-cell standard deviations at masked
positions, and performs paired Wilcoxon + paired Cohen's d on the per-cell
stds. The module is intentionally minimal and intended to be imported and
called from your experiment driver.
"""
import sys
import os

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
import random
import torch

from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer


def paired_cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    n = len(diff)
    if n < 2:
        return np.nan
    md = np.nanmean(diff)
    sd = np.nanstd(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return md / sd


class UncertaintyAnalysis:
    """Minimal ablation comparing BGAIN and BN_AUG_Imputer."""

    def __init__(self, n_imputations: int = 30, random_seed: int = 42, debug: bool = False,
                 epochs: int = 50, test_size: float = 0.2):
        # number of stochastic imputations / samples per trained model
        self.n_imputations = int(n_imputations)
        self.random_seed = int(random_seed)
        self.debug = bool(debug)
        # number of epochs to train neural imputers per run (match tests/main.py)
        self.epochs = int(epochs)
        # store callables that create fresh instances per imputation run.
        # We intentionally create full imputer instances (not reuse objects)
        # to match the behaviour in tests/main.py and avoid state leakage.
        self.method_factories = {
            'BGAIN': lambda: BGAIN(epochs=self.epochs),
            'BN_AUG_Imputer': lambda: BN_AUG_Imputer(epochs=self.epochs)
        }
        # fraction of data held out as test for injecting missingness
        self.test_size = float(test_size)

    def _sample_from_trained(self, method, X_test_missing: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
        """
        Given a trained `method`, generate `self.n_imputations` stochastic
        imputed datasets for the provided `X_test_missing`. Prefer to use
        `method.model.sample_conditionally` when available (BN_AUG). Returns
        an array of shape (n_imputations, n_rows, n_cols).
        """
        samples = []
        for i in range(self.n_imputations):
            seed = int(self.random_seed) + i
            np.random.seed(seed)
            random.seed(seed)
            try:
                torch.manual_seed(seed)
            except Exception:
                pass

            Xm_run = X_test_missing.copy()

            # Prefer direct sampler if present on model
            sampled = None
            model = getattr(method, 'model', None)
            try:
                if model is not None and hasattr(model, 'sample_conditionally'):
                    sampled = model.sample_conditionally(Xm_run.copy(), Xm_run.isnull())
                elif hasattr(method, 'sdg_impute'):
                    # sdg_impute may average multiple samples; request n_iter=1
                    sampled = method.sdg_impute(Xm_run.copy(), n_iter=1, refine_passes=1)
                elif hasattr(method, 'impute_all_missing'):
                    sampled = method.impute_all_missing(Xm_run.copy())
                elif hasattr(method, 'transform'):
                    sampled = method.transform(Xm_run.copy())
            except Exception:
                sampled = None

            # If sampler returned nothing usable, fallback to instantiating a fresh method
            if sampled is None:
                try:
                    # Attempt to create a fresh instance, fit on the already-prepared training data
                    fresh = self.method_factories.get(type(method).__name__, None)
                    # If we cannot construct via factory by name, try the class directly
                    if fresh is None and hasattr(method, '__class__'):
                        fresh = lambda: method.__class__(epochs=getattr(method, 'epochs', self.epochs))
                    newm = fresh() if fresh is not None else None
                    if newm is not None and hasattr(newm, 'fit') and hasattr(method, '_fitted_on'):
                        # If original method saved metadata about training data, try to reuse; else skip
                        newm.fit(method._fitted_on)
                    elif newm is not None and hasattr(newm, 'fit'):
                        # we can't refit here because we don't have access to training set in this helper
                        pass
                    if newm is not None and hasattr(newm, 'impute_all_missing'):
                        sampled = newm.impute_all_missing(Xm_run.copy())
                except Exception:
                    sampled = None

            if sampled is None:
                # As a last resort use the baseline filled version (no variability)
                sampled = Xm_run.copy()

            if not isinstance(sampled, pd.DataFrame):
                sampled = pd.DataFrame(sampled, columns=X_test_missing.columns, index=X_test_missing.index)
            sampled = sampled.reindex(columns=X_test_missing.columns, index=X_test_missing.index)

            # Ensure masked NaNs are filled with baseline so we have numeric values
            sampled_where_masked = pd.DataFrame(np.nan, index=X_test_missing.index, columns=X_test_missing.columns)
            masked_vals = sampled.where(mask)
            filled_masked = masked_vals.fillna(X_test_missing.where(mask))
            sampled_where_masked[mask] = filled_masked[mask]

            samples.append(sampled_where_masked.values.astype(float))

        if not samples:
            return np.empty((0, X_test_missing.shape[0], X_test_missing.shape[1]))
        return np.stack(samples, axis=0)

    def _apply_mcar(self, X: pd.DataFrame, missing_rate: float, exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.RandomState(self.random_seed)
        Xm = X.copy()
        if exclude_columns is None:
            exclude_columns = []
        mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        cols = [c for c in X.columns if c not in exclude_columns]
        total = int(np.floor(missing_rate * X.shape[0] * len(cols)))
        if total <= 0:
            return Xm, mask
        choices = rng.choice(len(cols) * X.shape[0], size=total, replace=False)
        for idx in choices:
            r = idx // len(cols)
            c = cols[idx % len(cols)]
            Xm.iat[r, X.columns.get_loc(c)] = np.nan
            mask.iat[r, X.columns.get_loc(c)] = True
        return Xm, mask

    def _collect_imputations(self, method_name: str, X_train: pd.DataFrame, X_missing: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
        imputations = []
        factory = self.method_factories[method_name]
        for i in tqdm(range(self.n_imputations), desc=f"{method_name} imputations", leave=False):
            # set all RNG seeds so runs are reproducible and independent
            seed = int(self.random_seed) + int(i)
            np.random.seed(seed)
            random.seed(seed)
            try:
                torch.manual_seed(seed)
            except Exception:
                # torch may not be available in all environments; ignore if not
                pass

            Xm_run = X_missing.copy()
            # create a fresh imputer instance for this run (prevents state leakage)
            try:
                # factory may be a callable that returns a new instance
                if callable(factory):
                    method = factory()
                elif isinstance(factory, tuple) and len(factory) == 2:
                    cls, kwargs = factory
                    method = cls(**kwargs)
                else:
                    # try to call/instantiate whatever was provided
                    method = factory()
            except Exception as e:
                if self.debug:
                    print(f"Could not create imputer instance for {method_name} on run {i}: {e}")
                # skip this imputation run
                continue

            try:
                # Train on the clean baseline to avoid training failures (BGAIN requires no nulls)
                if hasattr(method, 'fit'):
                    method.fit(X_train)

                if hasattr(method, 'impute_all_missing'):
                    imputed = method.impute_all_missing(Xm_run.copy())
                elif hasattr(method, 'transform'):
                    imputed = method.transform(Xm_run.copy())
                else:
                    imputed = Xm_run.copy()

                if not isinstance(imputed, pd.DataFrame):
                    imputed = pd.DataFrame(imputed, columns=X_train.columns, index=X_train.index)

                # Ensure rows/columns align with training data (important: index and columns)
                imputed = imputed.reindex(index=X_train.index, columns=X_train.columns)

                # Quick helper to count filled masked cells using pandas (robust to dtypes)
                def count_filled_masked(df):
                    df2 = df.reindex(index=X_train.index, columns=X_train.columns)
                    arr = df2.values
                    mask_arr = mask.values
                    # Use numpy boolean indexing when shapes align to avoid
                    # pandas alignment/broadcasting quirks that inflate counts.
                    if arr.shape == mask_arr.shape:
                        try:
                            return int(np.count_nonzero(~pd.isna(arr[mask_arr])))
                        except Exception:
                            pass
                    # Fallback to pandas counting
                    return int(df2.where(mask).notna().sum().sum())

                filled_count = count_filled_masked(imputed)
                if self.debug:
                    print(f"Imputation {i} for {method_name}: {filled_count}/{int(mask.values.sum())} masked cells filled by primary imputer.")

                # If imputer returned no filled masked cells, try reasonable
                # fallback calls specific to the imputer class. This mirrors
                # the behaviour in `tests/main.py` where methods are called
                # via a unified `impute_all_missing` interface, but here we
                # attempt alternative public APIs before giving up.
                if filled_count == 0:
                    if method_name == 'BN_AUG_Imputer' and hasattr(method, 'sdg_impute'):
                        if self.debug:
                            print(f"Primary BN_AUG impute returned no values; trying sdg_impute fallback (single sample).")
                        try:
                            alt = method.sdg_impute(Xm_run.copy(), n_iter=1, refine_passes=1)
                            if not isinstance(alt, pd.DataFrame):
                                alt = pd.DataFrame(alt, columns=X_train.columns, index=X_train.index)
                            alt = alt.reindex(columns=X_train.columns)
                            if count_filled_masked(alt) > 0:
                                imputed = alt
                                filled_count = count_filled_masked(imputed)
                                if self.debug:
                                    print(f"sdg_impute provided {filled_count} filled masked cells.")
                        except Exception as e:
                            if self.debug:
                                print(f"sdg_impute fallback failed: {e}")

                    # Additional non-invasive fallback: if sdg_impute/transform
                    # returned no filled masked cells, try calling the underlying
                    # BN_AUG model sampler directly. This requests a single
                    # stochastic sample from the trained BN-AUG SDG and maps it
                    # through the transformer's inverse. It avoids editing the
                    # imputer internals and mirrors what sdg_impute does.
                    if filled_count == 0 and method_name == 'BN_AUG_Imputer':
                        try:
                            # Prefer method.model.sample_conditionally if available
                            model = getattr(method, 'model', None)
                            if model is not None and hasattr(model, 'sample_conditionally'):
                                if self.debug:
                                    print(f"Trying model.sample_conditionally fallback for {method_name} (single stochastic sample).")
                                sampled = model.sample_conditionally(Xm_run.copy(), Xm_run.isnull())
                                if not isinstance(sampled, pd.DataFrame):
                                    sampled = pd.DataFrame(sampled, columns=X_train.columns, index=X_train.index)
                                sampled = sampled.reindex(columns=X_train.columns)
                                if count_filled_masked(sampled) > 0:
                                    imputed = sampled
                                    filled_count = count_filled_masked(imputed)
                                    if self.debug:
                                        print(f"model.sample_conditionally provided {filled_count} filled masked cells.")
                        except Exception as e:
                            if self.debug:
                                print(f"model.sample_conditionally fallback failed: {e}")

                    if filled_count == 0 and hasattr(method, 'transform'):
                        if self.debug:
                            print(f"Trying `transform` fallback for {method_name}.")
                        try:
                            alt = method.transform(Xm_run.copy())
                            if not isinstance(alt, pd.DataFrame):
                                alt = pd.DataFrame(alt, columns=X_train.columns, index=X_train.index)
                            alt = alt.reindex(columns=X_train.columns)
                            if count_filled_masked(alt) > 0:
                                imputed = alt
                                filled_count = count_filled_masked(imputed)
                                if self.debug:
                                    print(f"transform provided {filled_count} filled masked cells.")
                        except Exception as e:
                            if self.debug:
                                print(f"transform fallback failed: {e}")

                # As a last resort we will fill masked NaNs with baseline
                # values so we end up with numeric inputs for std/mean.
                imputed_where_masked = pd.DataFrame(np.nan, index=X_train.index, columns=X_train.columns)
                masked_vals = imputed.where(mask)
                n_masked = int(mask.values.sum())
                # Count NaNs only at masked positions after fallbacks; prefer
                # numpy boolean indexing when possible.
                try:
                    arr = imputed.reindex(index=X_train.index, columns=X_train.columns).values
                    mask_arr = mask.values
                    if arr.shape == mask_arr.shape:
                        n_masked_nan = int(np.count_nonzero(pd.isna(arr)[mask_arr]))
                    else:
                        n_masked_nan = int(masked_vals.isna().sum().sum())
                except Exception:
                    n_masked_nan = int(masked_vals.isna().sum().sum())
                if n_masked_nan > 0 and self.debug:
                    print(f"After fallbacks, {n_masked_nan}/{n_masked} masked cells are NaN; filling from baseline.")
                filled_masked = masked_vals.fillna(X_train.where(mask))
                imputed_where_masked[mask] = filled_masked[mask]

                # Save one diagnostic CSV for the first imputation of each method
                if self.debug and i == 0:
                    try:
                        imputed_where_masked.to_csv(f"debug_imputed_{method_name}_run{i}.csv", index=False)
                        if self.debug:
                            print(f"Wrote debug_imputed_{method_name}_run{i}.csv")
                    except Exception:
                        pass

                imputations.append(imputed_where_masked.values.astype(float))
            except Exception as e:
                if self.debug:
                    print(f"Imputation {i} failed for {method_name}: {e}")
                continue

        if not imputations:
            return np.empty((0, X_train.shape[0], X_train.shape[1]))
        return np.stack(imputations, axis=0)

    def analyze(self, X: pd.DataFrame, missing_rate: float = 0.1, exclude_columns: List[str] = None) -> Dict:
        if exclude_columns is None:
            exclude_columns = []

        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].replace('?', np.nan)
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                pass

        baseline_imputer = SimpleImputer(strategy='mean')
        X_clean = pd.DataFrame(baseline_imputer.fit_transform(X), columns=X.columns, index=X.index)

        X_missing, mask = self._apply_mcar(X_clean, missing_rate, exclude_columns=exclude_columns)

        collected = {}
        for m in ['BGAIN', 'BN_AUG_Imputer']:
            if self.debug:
                print(f"Collecting imputations for {m}...")
            arr = self._collect_imputations(m, X_clean, X_missing, mask)
            collected[m] = arr

        # Diagnostic dumps: save raw collected arrays and a short summary to
        # help debug cases where a method shows zero variability.
        if self.debug:
            for m, arr in collected.items():
                try:
                    if arr.size == 0:
                        print(f"[diag] {m}: no imputations collected")
                        continue
                    # Save stacked imputations for offline inspection
                    np.savez_compressed(f"debug_collected_{m}.npz", arr=arr)

                    # Per-run fraction of masked cells that are non-NaN
                    n_runs = arr.shape[0]
                    n_cells = arr.shape[1] * arr.shape[2]
                    per_run_fill = []
                    for i in range(n_runs):
                        a = arr[i]
                        # Only consider originally-masked positions
                        masked_vals = a.reshape(-1)[mask.values.reshape(-1)]
                        filled = np.count_nonzero(~np.isnan(masked_vals))
                        per_run_fill.append(filled / float(mask.values.sum()))

                    per_run_fill = np.array(per_run_fill)
                    all_identical = all(np.allclose(arr[0], arr[i], atol=1e-12, equal_nan=True) for i in range(1, n_runs))
                    per_cell_var = np.nanvar(arr, axis=0)
                    masked_var = per_cell_var[mask.values]
                    print(f"[diag] {m}: runs={n_runs}, masked_cells={int(mask.values.sum())}, "
                          f"per-run-fill_mean={per_run_fill.mean():.3f}, per-run-fill_min={per_run_fill.min():.3f}, "
                          f"per-run-fill_max={per_run_fill.max():.3f}, all_identical={all_identical}, "
                          f"masked_var_mean={np.nanmean(masked_var):.6g}, masked_var_max={np.nanmax(masked_var):.6g}")
                except Exception as e:
                    print(f"[diag] failed for {m}: {e}")

        # No fallback here: rely on raw imputer outputs. If BN_AUG still
        # produces identical/zero variability we should inspect the
        # imputer's implementation or saved diagnostics. Keeping the
        # harness minimal mirrors tests/main.py behaviour.

        results = {}
        for m, arr in collected.items():
            if arr.size == 0:
                results[m] = {'stds': np.array([]), 'means': np.array([]), 'n_imputations': 0}
                continue
            # suppress runtime warnings about mean/std of empty slices
            with np.errstate(invalid='ignore', divide='ignore'):
                std_map = np.nanstd(arr, axis=0)
                mean_map = np.nanmean(arr, axis=0)
            stds = std_map[mask.values]
            means = mean_map[mask.values]
            valid_mask = ~np.isnan(stds)
            results[m] = {'stds': stds[valid_mask], 'means': means[valid_mask], 'n_imputations': arr.shape[0]}

        s1 = results['BGAIN']['stds']
        s2 = results['BN_AUG_Imputer']['stds']
        min_len = min(len(s1), len(s2))
        paired = {}
        if min_len == 0:
            paired['wilcoxon_p'] = np.nan
            paired['cohens_d'] = np.nan
            paired['n_pairs'] = 0
        else:
            x = s1[:min_len]
            y = s2[:min_len]
            try:
                stat, p = wilcoxon(x, y)
            except Exception:
                p = np.nan
            d = paired_cohens_d(x, y)
            paired['wilcoxon_p'] = p
            paired['cohens_d'] = d
            paired['n_pairs'] = min_len

        out_df = pd.DataFrame({
            'BGAIN_std': np.pad(results['BGAIN']['stds'], (0, max(0, len(results['BN_AUG_Imputer']['stds'])-len(results['BGAIN']['stds']))), constant_values=np.nan),
            'BN_AUG_std': np.pad(results['BN_AUG_Imputer']['stds'], (0, max(0, len(results['BGAIN']['stds'])-len(results['BN_AUG_Imputer']['stds']))), constant_values=np.nan)
        })
        out_df.to_csv('uncertainty_per_cell_stds.csv', index=False)

        try:
            plt.figure(figsize=(6, 4))
            data = []
            if results['BGAIN']['stds'].size > 0:
                data.append(pd.DataFrame({'std': results['BGAIN']['stds'], 'method': 'BGAIN'}))
            if results['BN_AUG_Imputer']['stds'].size > 0:
                data.append(pd.DataFrame({'std': results['BN_AUG_Imputer']['stds'], 'method': 'BN_AUG_Imputer'}))
            if data:
                import seaborn as sns
                dfp = pd.concat(data, ignore_index=True)
                sns.boxplot(data=dfp, x='method', y='std')
                plt.title('Per-cell std (uncertainty)')
                plt.tight_layout()
                plt.savefig('uncertainty_boxplot.png')
            plt.close()
        except Exception:
            pass

        summary = {
            'n_imputations_BGAIN': results['BGAIN']['n_imputations'],
            'n_imputations_BN_AUG_Imputer': results['BN_AUG_Imputer']['n_imputations'],
            'paired': paired,
            'per_cell_csv': 'uncertainty_per_cell_stds.csv'
        }
        return summary


if __name__ == '__main__':
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description='Run focused uncertainty ablation (BGAIN vs BN_AUG_Imputer)')
    parser.add_argument('--csv', type=str, default=None, help='Path to a CSV file to load as DataFrame (optional)')
    parser.add_argument('--demo', action='store_true', help='Run a small synthetic demo instead of loading a file')
    parser.add_argument('--n-imputations', type=int, default=5, help='Number of stochastic imputations per method')
    parser.add_argument('--missing-rate', type=float, default=0.1, help='MCAR missing rate to apply')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exclude', type=str, default='', help='Comma-separated columns to exclude from masking (e.g., target)')
    parser.add_argument('--out-prefix', type=str, default='uncertainty', help='Prefix for output files')
    args = parser.parse_args()

    # Load or synthesize data
    df = None
    if args.csv:
        p = os.path.expanduser(args.csv)
        if not os.path.exists(p):
            print(f"CSV path not found: {p}", file=sys.stderr)
            sys.exit(2)
        try:
            # try a flexible CSV read; comment='%' allows ARFF-style comments in CSV exports
            df = pd.read_csv(p, comment='%', header=0)
        except Exception as e:
            print(f"Failed to read CSV at {p}: {e}", file=sys.stderr)
            sys.exit(3)

    if df is None and not args.demo:
        # Convenience: if no CSV was provided, run the synthetic demo by default
        print("No CSV provided; running synthetic demo (same as --demo).", file=sys.stderr)
        args.demo = True

    if df is None and args.demo:
        # Synthetic numeric dataset (fast, deterministic with seed)
        rng = np.random.RandomState(args.seed)
        n_rows = 200
        n_cols = 8
        data = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_cols))
        cols = [f'feat{i}' for i in range(n_cols)]
        df = pd.DataFrame(data, columns=cols)

    exclude_columns = [c.strip() for c in args.exclude.split(',') if c.strip()]

    analyzer = UncertaintyAnalysis(n_imputations=args.n_imputations, random_seed=args.seed, debug=True)
    print(f"Running ablation: BGAIN vs BN_AUG_Imputer (n_imputations={args.n_imputations}, missing_rate={args.missing_rate})")
    try:
        summary = analyzer.analyze(df, missing_rate=args.missing_rate, exclude_columns=exclude_columns)
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        raise

    # Print paired results if present
    paired = summary.get('paired', {})
    print('\n=== Paired comparison (per-cell stds) ===')
    print(f"Wilcoxon p-value: {paired.get('wilcoxon_p', float('nan'))}")
    print(f"Paired Cohen's d: {paired.get('cohens_d', float('nan'))}")
    print(f"Number of paired cells: {paired.get('n_pairs', 0)}")

    # Move generated outputs to names with prefix
    csv_src = 'uncertainty_per_cell_stds.csv'
    png_src = 'uncertainty_boxplot.png'
    csv_dst = f"{args.out_prefix}_per_cell_stds.csv"
    png_dst = f"{args.out_prefix}_boxplot.png"
    try:
        if os.path.exists(csv_src):
            os.replace(csv_src, csv_dst)
        if os.path.exists(png_src):
            os.replace(png_src, png_dst)
        print(f"Saved outputs: {csv_dst}, {png_dst}")
    except Exception:
        print("Could not move output files; they remain in the working directory if created.")

    print('\nDone.')
