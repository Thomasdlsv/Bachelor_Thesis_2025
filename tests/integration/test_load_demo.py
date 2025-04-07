from bgan.synthesizers.bgan import BGAN
from bgan import load_demo
import numpy as np
from scipy import stats

def test_load_demo():
    """End-to-end test to load and synthesize data."""
    # Setup
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income',
    ]
    print("\nInitializing BGAN model...")
    ctgan = BGAN(epochs=1)

    # Run
    print("Loading demo data...")
    data = load_demo()
    real_data = data.copy()  # Assign real_data to the original dataset
    print(f"Original data shape: {data.shape}")
    print("\nTraining model...")
    ctgan.fit(data, discrete_columns)
    print("\nGenerating synthetic samples...")
    synthetic_data = ctgan.sample(1000)
    #samples = ctgan.sample(1000, condition_column='native-country', condition_value='United-States')
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print("\nFirst few rows of synthetic data:")
    print(synthetic_data.head())

    # Validation checks
    print("\nValidation Results:")
    
    # 1. Check column consistency
    print("\nColumn matching:", set(real_data.columns) == set(synthetic_data.columns))
    
    # 2. Check categorical distributions
    for col in discrete_columns:
        real_dist = real_data[col].value_counts(normalize=True)
        syn_dist = synthetic_data[col].value_counts(normalize=True)
        print(f"\n{col} category distribution similarity:", 
              1 - np.mean(np.abs(real_dist - syn_dist)))
    
    # 3. Check continuous columns
    continuous_cols = [c for c in real_data.columns if c not in discrete_columns]
    for col in continuous_cols:
        ks_stat, p_val = stats.ks_2samp(real_data[col], synthetic_data[col])
        print(f"\n{col} KS test p-value:", p_val)
        print(f"{col} mean difference:", 
              abs(real_data[col].mean() - synthetic_data[col].mean()))

    # Assert
    #assert samples.shape == (1000, 15)
    #assert all([col[0] != ' ' for col in samples.columns])
    #assert not samples.isna().any().any()

if __name__ == "__main__":
    test_load_demo()