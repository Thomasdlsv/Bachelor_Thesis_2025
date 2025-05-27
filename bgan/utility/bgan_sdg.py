from bgan.synthesizers.bgan import BGAN
import pandas as pd

class BGAN_SDG:
    """
    Wrapper class for the BGAN (Bayesian Generative Adversarial Network) synthesizer.
    Provides a simplified interface for fitting on real data and sampling synthetic data.
    """
    def __init__(self, epochs, use_uncertainty_loss=True, use_kl_loss=True, optimizer_type="adam"):
        """
        Initialize the BGAN_SDG synthesizer.

        Args:
            epochs (int): Number of training epochs.
            use_uncertainty_loss (bool): Whether to use uncertainty loss during training.
            use_kl_loss (bool): Whether to use KL divergence loss during training.
            optimizer_type (str): Optimizer type for training (e.g., "adam").
        """
        self.bgan = BGAN(
            epochs=epochs,
            use_uncertainty_loss=use_uncertainty_loss,
            use_kl_loss=use_kl_loss,
            optimizer_type=optimizer_type
        )

    def fit(self, real_data, discrete_columns):
        """
        Fit the BGAN model on real data.

        Args:
            real_data (pd.DataFrame): The real dataset to train on.
            discrete_columns (list): List of discrete/categorical column names.
        """
        self.bgan.fit(real_data, discrete_columns)

    def sample(self, n_samples):
        """
        Generate synthetic samples from the trained BGAN model.

        Args:
            n_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Synthetic data samples.
        """
        return self.bgan.sample(n_samples)  