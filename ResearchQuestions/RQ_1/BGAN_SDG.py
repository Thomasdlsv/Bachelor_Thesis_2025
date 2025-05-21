from bgan.synthesizers.bgan import BGAN
import pandas as pd

class BGAN_SDG:
    def __init__(self, epochs):
        self.bgan = BGAN(epochs=epochs)

    def fit(self, real_data, discrete_columns):
        self.bgan.fit(real_data, discrete_columns)

    def sample(self, n_samples):
        return self.bgan.sample(n_samples)


    # Load data
    # data = pd.read_csv("http://ctgan-demo.s3.amazonaws.com/census.csv.gz")  
    # discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # # Generate synthetic data
    # print("\nTesting BGAN...")
    # bgan_sdg = BGAN_SDG(epochs=1)
    # bgan_sdg.bgan.fit(data, discrete_columns)
    # synthetic_data = bgan_sdg.bgan.sample(len(data))

    # # Evaluate quality
    # metrics = bgan_sdg.evaluate_quality(data, synthetic_data)
    
    # # Print metrics
    # print("\nPerformance Metrics:")
    # print("-" * 50)
    # print(f"ML Performance:")
    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1']:.4f}")
    
    # print("\nStatistical Similarity (Wasserstein distance):")
    # for key, value in metrics.items():
    #     if '_wasserstein' in key:
    #         print(f"{key}: {value:.4f}")

    # # Visualize distributions
    # plotter = DataComparisonPlotter(real_data=data, synthetic_data=synthetic_data)
    # plotter.plot_distributions()
    # plotter.plot_tsne()