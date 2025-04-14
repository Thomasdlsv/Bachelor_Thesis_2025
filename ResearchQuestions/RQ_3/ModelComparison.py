import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from bgan.synthesizers.bgan import BGAN
from bgan import load_demo
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import seaborn as sns


class ModelComparison:
    def __init__(self):
        self.discrete_columns = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country', 'income'
        ]
        self.data = self.preprocess_data(load_demo())
        self.numeric_data = self.data.select_dtypes(include=[np.number])
        self.results = {}

        print("\nLoaded Data:")
        print(self.data.head())
        print("\nNumeric Data:")
        print(self.numeric_data.head())

    def preprocess_data(self, data):
        data = data.replace('?', np.nan)
        data = data.fillna(0)
        return data

    def ks_test(self, real_data, synthetic_data):
        scores = {}
        for col in real_data.columns:
            if np.issubdtype(real_data[col].dtype, np.number):
                statistic, p_value = ks_2samp(real_data[col], synthetic_data[col])
                scores[col] = {'ks_stat': statistic, 'p_value': p_value}
        return scores

    def jsd_for_column(self, real, synthetic, bins=20):
        r_hist, _ = np.histogram(real, bins=bins, range=(min(real.min(), synthetic.min()), max(real.max(), synthetic.max())), density=True)
        s_hist, _ = np.histogram(synthetic, bins=bins, range=(min(real.min(), synthetic.min()), max(real.max(), synthetic.max())), density=True)

        r_hist += 1e-8
        s_hist += 1e-8
        r_hist /= r_hist.sum()
        s_hist /= s_hist.sum()

        return jensenshannon(r_hist, s_hist)

    def test_bgan(self):
        print("\nTesting BGAN...")
        bgan = BGAN(epochs=100) #changing epochs for testing
        bgan.fit(self.data, self.discrete_columns)
        synthetic_data_bgan = bgan.sample(len(self.data))

        ks_results = self.ks_test(self.numeric_data, synthetic_data_bgan[self.numeric_data.columns])
        jsd_results = {
            col: self.jsd_for_column(self.numeric_data[col], synthetic_data_bgan[col])
            for col in self.numeric_data.columns
        }

        return ks_results, jsd_results, synthetic_data_bgan[self.numeric_data.columns]

    def test_ctgan(self):
        print("\nTesting CTGAN...")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data)

        ctgan = CTGANSynthesizer(metadata, epochs=100)
        ctgan.fit(self.data)
        synthetic_data_ctgan = ctgan.sample(len(self.data))

        ks_results = self.ks_test(self.numeric_data, synthetic_data_ctgan[self.numeric_data.columns])
        jsd_results = {
            col: self.jsd_for_column(self.numeric_data[col], synthetic_data_ctgan[col])
            for col in self.numeric_data.columns
        }

        return ks_results, jsd_results, synthetic_data_ctgan[self.numeric_data.columns]


    def compare_models(self):
        ks_results_bgan, jsd_results_bgan, bgan_numeric = self.test_bgan()
        ks_results_ctgan, jsd_results_ctgan, ctgan_numeric = self.test_ctgan()

        print("\nComparison Summary:")
        print("-" * 50)
        for model_name, ks_results, jsd_results in [('BGAN', ks_results_bgan, jsd_results_bgan),
                                                    ('CTGAN', ks_results_ctgan, jsd_results_ctgan)]:
            print(f"\n{model_name.upper()} METRICS:")
            print("Kolmogorov–Smirnov (KS) Test:")
            for col, val in ks_results.items():
                print(f"  {col}: KS = {val['ks_stat']:.4f}, p = {val['p_value']:.4f}")
            print("Jensen-Shannon Divergence (JSD):")
            for col, jsd in jsd_results.items():
                print(f"  {col}: JSD = {jsd:.4f}")

        return ks_results_bgan, ks_results_ctgan, jsd_results_bgan, jsd_results_ctgan, bgan_numeric, ctgan_numeric


def plot_comparison_metrics(ks_results_bgan, ks_results_ctgan, jsd_results_bgan, jsd_results_ctgan, columns):
    ks_bgan = [ks_results_bgan[col]['ks_stat'] for col in columns]
    ks_ctgan = [ks_results_ctgan[col]['ks_stat'] for col in columns]
    jsd_bgan = [jsd_results_bgan[col] for col in columns]
    jsd_ctgan = [jsd_results_ctgan[col] for col in columns]

    x = np.arange(len(columns))
    width = 0.35

    # KS Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, ks_bgan, width, label='BGAN KS', color='b')
    ax.bar(x + width/2, ks_ctgan, width, label='CTGAN KS', color='g')
    ax.set_xlabel('Features')
    ax.set_ylabel('KS Value')
    ax.set_title('Kolmogorov-Smirnov (KS) Test Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # JSD Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, jsd_bgan, width, label='BGAN JSD', color='orange')
    ax.bar(x + width/2, jsd_ctgan, width, label='CTGAN JSD', color='r')
    ax.set_xlabel('Features')
    ax.set_ylabel('JSD Value')
    ax.set_title('Jensen-Shannon Divergence (JSD) Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(real_df, synthetic_df_bgan, synthetic_df_ctgan, features, bins=50):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(real_df[feature], label='Real', linewidth=2)
        sns.kdeplot(synthetic_df_bgan[feature], label='BGAN', linestyle='--')
        sns.kdeplot(synthetic_df_ctgan[feature], label='CTGAN', linestyle=':')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ✅ Main Entry Point
if __name__ == "__main__":
    comparison = ModelComparison()
    ks_bgan, ks_ctgan, jsd_bgan, jsd_ctgan, bgan_numeric, ctgan_numeric = comparison.compare_models()

    numeric_data = comparison.numeric_data
    numeric_cols = numeric_data.columns.tolist()

    plot_comparison_metrics(ks_bgan, ks_ctgan, jsd_bgan, jsd_ctgan, numeric_cols)

    features_to_plot = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
    plot_feature_distributions(
        real_df=numeric_data,
        synthetic_df_bgan=bgan_numeric,
        synthetic_df_ctgan=ctgan_numeric,
        features=features_to_plot
    )
