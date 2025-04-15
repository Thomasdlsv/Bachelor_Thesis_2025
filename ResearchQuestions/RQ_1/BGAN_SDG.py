from ResearchQuestions.RQ_1.DataComparisonPlotter import DataComparisonPlotter
from bgan.synthesizers.bgan import BGAN
import pandas as pd

class BGAN_SDG:

    if __name__ == "__main__":
        data = pd.read_csv("http://ctgan-demo.s3.amazonaws.com/census.csv.gz")  # Replace with your dataset
        discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Visualize synthetic vs real data
        print("\nTesting BGAN...")
        bgan = BGAN(epochs=1) #changing epochs for testing
        bgan.fit(data, discrete_columns)
        synthetic_data_bgan = bgan.sample(len(data))

        plotter = DataComparisonPlotter(real_data=data, synthetic_data=synthetic_data_bgan)
        plotter.plot_distributions()     # Histograms/KDEs
        plotter.plot_tsne()              # t-SNE visualization