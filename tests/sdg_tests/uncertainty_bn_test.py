import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import f_oneway
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from scipy.stats import kruskal, sem


from bn_bgan.bn_bgan_sdg import BN_AUG_SDG

# ======================================================================================================================
# Main execution block to run the evaluation between bn influence on uncertainty, checking for statistical significance
# ======================================================================================================================

if __name__ == "__main__":
    '''
    This script evaluates the impact of different bayesian network influences on the uncertainty of synthetic data generated
    by the BN-AUG-SDG model. It runs multiple trials with varying parameters, computes the mean variance of the synthetic data,
    and checks for convergence using the standard error of the mean (SEM). Finally, it performs a statistical test to determine
    if there are significant differences in uncertainty across different batch normalization influences.
    '''

    #Load real data and preprocess
    real_data = pd.read_csv("http://ctgan-demo.s3.amazonaws.com/census.csv.gz")
    discrete_columns = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
    real_train, real_eval = train_test_split(real_data, test_size=0.3, random_state=42)
    
    # ==================
    # TESTING PARAMETERS
    # ==================
    MAX_RUNS = 4
    N_SAMPLES = 1000
    SIGNIFICANCE_THRESHOLD = 0.05

    bnaug_param_grid = {
         'epochs': [1, 25, 50],
         #'batch_norm': [True, False],
         #'use_uncertainty_loss': [True, False],
         #'use_kl_loss': [True, False],
         'optimizer_type': ["adam", "adamw", "rmsprop"], 
         #'bn_influence': [0.1, 0.5, 0.9]
    }

    # compare uncertainty metrics for different bn_influence values
    bnaug_uncertainty = []
    for params in ParameterGrid(bnaug_param_grid):
        mean_uncertainties = []
        for seed in range(MAX_RUNS):
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
            mean_uncertainties.append(mean_uncertainty)
            # read bn_influence safely (ParameterGrid may not include it)
            bn_inf = params.get("bn_influence", 0.0)
            bnaug_uncertainty.append({
                "bn_influence": bn_inf,
                "mean_uncertainty": mean_uncertainty,
                "seed": seed
            })
            print(f"Mean variance (uncertainty): {mean_uncertainty:.6f} (bn_influence={bn_inf})")

            # Check convergence
            if len(mean_uncertainties) >= 5:
                current_sem = sem(mean_uncertainties)
                print(f"Current SEM: {current_sem:.6f}")
                if current_sem < SIGNIFICANCE_THRESHOLD:
                    print(f"Converged for bn_influence={params['bn_influence']} after {seed+1} runs (SEM={current_sem:.6f})")
                    break

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
    # handle missing bn_influence gracefully by filling NaNs with a string label
    if uncertainty_df["bn_influence"].isnull().any():
        uncertainty_df["bn_influence"] = uncertainty_df["bn_influence"].fillna("none")

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