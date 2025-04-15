from bayesian_network.DataSamplerBN import ColumnInfo
from bayesian_network.BayesianGANInference import BayesianGANInference
from bayesian_network.BayesianGANPipeline import BayesianGANPipeline
from ResearchQuestions.RQ_1.DataComparisonPlotter import DataComparisonPlotter
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

class BN_AUG_SDG:
    pass

if __name__ == "__main__":
    

    # Example dataset
    data = pd.read_csv("http://ctgan-demo.s3.amazonaws.com/census.csv.gz")  
    discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode categorical values
    le_dict = {}
    for col in discrete_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le

    # Define output_info
    output_info = [ColumnInfo(dim=1, activation_fn='softmax' if col in discrete_columns else 'tanh') for col in data.columns]

    # Define Bayesian Network structure
    bn_structure = {
        0: {'parents': []},
        1: {'parents': [0]},
        2: {'parents': [1]},
        3: {'parents': [2]},
        4: {'parents': [3]},
    }

    # Initialize the pipeline
    pipeline = BayesianGANPipeline(
        data=data,
        output_info=output_info,
        bn_structure=bn_structure,
        discrete_columns=discrete_columns,
        epochs=50,
        batch_size=256
    )

    # Train the Bayesian GAN
    pipeline.train()

    # Generate synthetic data
    synthetic_data = pipeline.generate_synthetic_data(num_samples=1000)

    # Initialize BayesianGANInference
    f = lambda x: x  # Example forward model
    sigma_noise = np.eye(data.shape[1]) * 0.1
    inference = BayesianGANInference(
        generator=pipeline.bgan._generator,
        data_sampler_bn=pipeline.data_sampler,
        f=f,
        sigma_noise=sigma_noise
    )

    # Perform inference tasks
    y_hat = np.random.rand(data.shape[1])  # Example observation
    mc_expectation = inference.monte_carlo_expectation(y_hat, l_func=lambda x: x)
    print("Monte Carlo Expectation:", mc_expectation)

    # Visualize synthetic vs real data
    plotter = DataComparisonPlotter(real_data=data, synthetic_data=synthetic_data)
    plotter.plot_distributions()     # Histograms/KDEs
    plotter.plot_tsne()              # t-SNE visualization

    # Evaluate synthetic data accuracy
    for col in discrete_columns:
        real_col = le_dict[col].inverse_transform(data[col])
        synthetic_col = le_dict[col].inverse_transform(synthetic_data[col].astype(int))
        acc = accuracy_score(real_col, synthetic_col)
        print(f"Accuracy for column {col}: {acc:.4f}")

    # Evaluate numerical columns using MSE
    numerical_columns = [col for col in data.columns if col not in discrete_columns]
    for col in numerical_columns:
        mse = mean_squared_error(data[col], synthetic_data[col])
        print(f"Mean Squared Error for column {col}: {mse:.4f}")