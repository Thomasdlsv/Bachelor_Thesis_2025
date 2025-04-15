import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class DataComparisonPlotter:
    def __init__(self, real_data, synthetic_data):
        """
        Args:
            real_data (pd.DataFrame): The original dataset.
            synthetic_data (pd.DataFrame): The generated dataset.
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    def plot_distributions(self, columns=None, bins=30, kde=True):
        """
        Plot histograms/KDE for each selected column.

        Args:
            columns (list): List of column names to plot. Defaults to all.
            bins (int): Number of histogram bins.
            kde (bool): Include KDE overlay.
        """
        if columns is None:
            columns = self.real_data.columns

        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
        plt.figure(figsize=(6 * n_cols, 4 * n_rows))

        for i, col in enumerate(columns):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.real_data[col], label='Real', color='blue', bins=bins, kde=kde, stat='density', alpha=0.6)
            sns.histplot(self.synthetic_data[col], label='Synthetic', color='orange', bins=bins, kde=kde, stat='density', alpha=0.6)
            plt.title(col)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_tsne(self, perplexity=30, n_iter=1000, random_state=0):
        """
        Plot 2D t-SNE projection of real and synthetic datasets.

        Args:
            perplexity (int): t-SNE perplexity value.
            n_iter (int): Number of iterations.
            random_state (int): Seed.
        """
        combined = np.vstack([self.real_data.values, self.synthetic_data.values])
        labels = np.array(['Real'] * len(self.real_data) + ['Synthetic'] * len(self.synthetic_data))

        combined_scaled = StandardScaler().fit_transform(combined)

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_result = tsne.fit_transform(combined_scaled)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, alpha=0.6, palette=['blue', 'orange'])
        plt.title("t-SNE: Real vs Synthetic")
        plt.legend()
        plt.tight_layout()
        plt.show()
