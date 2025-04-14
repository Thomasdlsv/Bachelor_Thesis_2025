import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from ctgan import CTGAN

# ---- Demo Loader ----
DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'

def load_demo():
    """Load the demo census dataset."""
    return pd.read_csv(DEMO_URL, compression='gzip')


# ---- Imputation Comparator ----
class GANImputerComparator:
    def __init__(self, bgan_model):
        self.bgan = bgan_model
        self.ctgan = CTGAN(epochs=10)

        self.discrete_columns = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country', 'income'
        ]
        self.data = self.preprocess_data(load_demo())
        self.numeric_data = self.data.select_dtypes(include=[np.number])

        print("\nLoaded Data:")
        print(self.data.head())
        print("\nNumeric Data:")
        print(self.numeric_data.head())

    def preprocess_data(self, df):
        df = df.dropna().reset_index(drop=True)
        df_encoded = pd.get_dummies(df, columns=self.discrete_columns)
        return df_encoded

    def mask_data(self, data, missing_rate=0.1, random_state=42):
        np.random.seed(random_state)
        mask = np.random.rand(*data.shape) < missing_rate
        data_masked = data.copy()
        data_masked[mask] = np.nan
        return data_masked, mask

    def impute_bgan(self, masked_data):
        return self.bgan.impute(masked_data)

    def impute_ctgan(self, data_masked, original_data):
        complete_data = data_masked.dropna()
        self.ctgan.fit(complete_data)
        synth_data = self.ctgan.sample(len(data_masked))
        imputed_data = data_masked.copy()
        for col in data_masked.columns:
            imputed_data[col] = imputed_data[col].fillna(synth_data[col])
        return imputed_data

    def evaluate(self, original, imputed, mask):
        true_vals = original.values[mask]
        pred_vals = imputed.values[mask]
        return np.sqrt(mean_squared_error(true_vals, pred_vals))

    def run(self, missing_rate=0.1):
        print("Masking data...")
        data_masked, mask = self.mask_data(self.numeric_data, missing_rate)

        print("Imputing with BGAN...")
        imputed_bgan = self.impute_bgan(data_masked)
        rmse_bgan = self.evaluate(self.numeric_data, imputed_bgan, mask)

        print("Imputing with CTGAN...")
        imputed_ctgan = self.impute_ctgan(data_masked, self.numeric_data)
        rmse_ctgan = self.evaluate(self.numeric_data, imputed_ctgan, mask)

        print(f"\nEvaluation Results:")
        print(f" - BGAN RMSE: {rmse_bgan:.4f}")
        print(f" - CTGAN RMSE: {rmse_ctgan:.4f}")

        print("\nExample Imputed Data (First 5 rows):")
        print("\nBGAN Imputation Sample:")
        print(imputed_bgan.head())  # Show imputed BGAN samples

        print("\nCTGAN Imputation Sample:")
        print(imputed_ctgan.head())  # Show imputed CTGAN samples

        return {
            'bgan_rmse': rmse_bgan,
            'ctgan_rmse': rmse_ctgan,
            'bgan_imputed': imputed_bgan,
            'ctgan_imputed': imputed_ctgan,
        }


# ---- Run if Main ----
if __name__ == "__main__":
    class DummyBGAN:
        def impute(self, data):
            # Simple mean imputer for now
            return data.fillna(data.mean(numeric_only=True))

    comparator = GANImputerComparator(bgan_model=DummyBGAN())
    results = comparator.run(missing_rate=0.2)
