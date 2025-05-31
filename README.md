# Synthetic Data Generation & Imputation  
**Maastricht University**  
**Bachelor of Data Science and Artificial Intelligence**  
**Author:** Thomás de los Santos Verrijp

This repository contains code for benchmarking and evaluating synthetic data generation (SDG) and imputation methods, including BGAN, BN-AUG-SDG, CTGAN, and others.

## Table of Contents
1. [Folder Setup](#folder-setup)
2. [Environment Setup](#environment-setup)  
3. [Running SDG Evaluations](#running-sdg-evaluations)  
4. [Running Imputation Evaluations](#running-imputation-evaluations)  
5. [Configurations](#configurations)  
6. [Customization and Advanced Usage](#customization-and-advanced-usage)
7. [Datasets](#datasets)
8. [Troubleshooting](#troubleshooting)

---

## 1. Folder Setup

/bgan: This folder contains the original implementation of the BGAN model (Bayesian Generative Adversarial Network). It includes core files that define how the model is built and trained. 

/bn_bgan: This contains my implementation of the Bayesian Network augmented GAN. The code here builds on the standard BGAN model and includes improvements such as batch normalization and better synthetic data quality control.

/datasets: This is where all the datasets used for experiments are stored. These datasets are primarily open-source healthcare datasets downloaded from OpenML. The key dataset used in imputation experiments is Fetal_Dataset.arff, but more datasets can be added here if needed.

/tests: This folder includes all scripts used to run experiments, evaluate performance, and generate visualizations for both synthetic data generation (SDG) and imputation.

---

## 2. Environment Setup

  Make sure you're in the directory of the project in your storage
    cd location
  
  Create an environment
    ```bash
    python -m venv venv
    ```
  
  Activate the environment
    ```bash
    source venv/bin/activate
    ```
  
  Make sure pip is up-to-date
    ```bash
    pip install --upgrade pip
    ```
  
  Download the requirements for this project:
    ```bash
    pip install -r requirements.txt
    ```

---
  
## 3. Running SDG Evaluations

  The main SDG evaluation script is:

  ```bash
  python -m tests.sdg_tests.main
  ```

  This script will:
  - Download and preprocess the Cancer_Dataset (can adjust which dataset loaded from the 'datasets' folder in the main code).
  - Train and evaluate BGAN, BN-AUG-SDG, CTGAN, and other models.
  - Output a DAG structure of the feature relationships everytime the BN-BGAN is trained
    --> To skip this, comment out the call to plot_bn_structure inside the main method in tests/sdg_tests/main.py
  - Output data distribution graphs for each feature in the dataset (and other miscellaneous visualisations) between BN-BGAN, BGAN, and the original dataset
    --> To skip this, comment out the call to the method 'evaluate_sdg' in tests/sdg_tests/main.py
  - Run a hyperparameter search comparing models with baselines, outputting performance metrics delineated in the report.
  - Currently, the dataset size is sampled in running, this is to ensure that the code is running as intended. When you can confirm it's running as it should, then comment out the sampling of the training and evaluation sets, and run the code again. 

---

## 4. Running Imputation Evaluations

  The main imputation script is:

  ```bash
  python -m tests.imputation_tests.main
  ```

  This script will:
  - Load and preprocess the Fetal_Dataset (can adjust which dataset loaded from the 'datasets' folder in the main code).
  - Define and configure several imputation methods (BGAIN, BN_AUG_Imputer, MICE, etc.).
  - Run repeated evaluation of imputation quality and downstream impact.
  - Aggregate and print results

---
   
## 5. Configurations
  - Experiment parameters: (number of repetitions, missing rates, random seed) can be set at the top of `tests/imputation_tests/main.py`.
  - Evaluation pipeline: missingness rates (10%, 20%,...) and missingness patterns (MCAR, MAR, MNAR) are handled in `tests/imputation_tests/configurations.py`.
  - You can add or remove imputation methods and adjust missingness patterns as needed.

---
  
## 6. Customization and Advanced Usage
  - All evaluation and visualization logic is modularized in the `SDGVisualizer` class (`tests/sdg_tests/sdg_visualizer.py`).
  - You can run only specific visualizations or metrics by calling the relevant methods from this class.
  - For hyperparameter search, adjust the parameter grids in the main scripts.
  - For new datasets, update the data loading section in the relevant script.

---

## 7. Datasets
  - All datasets used in this project are open-source and primarily sourced from the OpenML repository.
  - Datasets are classification-based and reside in the /datasets/ directory.
  - The focus of the experiments is on healthcare datasets. Only datasets relevant to healthcare should be used for benchmarking, as the report's findings and evaluations are based on this domain.
  - Additional datasets can be added by placing them into the /datasets/ folder. The code is modular and will handle them if the expected format is followed.
  - A larger, illustrative dataset (U.S. Census data) is included in the /bgan/utility/demo/ directory, and can also be accessed via:

  http://ctgan-demo.s3.amazonaws.com/census.csv.gz

  - This dataset serves primarily as a baseline for testing and for demonstrating the Bayesian Network (BN) structure visualizations. It is not used in the main experimental evaluations, but is useful for sanity-checking the BN-BGAN pipeline.

---

## 8. Troubleshooting
  - If you encounter missing package errors, ensure your environment is activated and run `pip install -r requirements.txt` again.
  - For large datasets or long runs, reduce the number of epochs or sample sizes in the configuration.
  - When running a hyperparameter search, ensure to output the text of only the parameters searched, otherwise can lead to errors.

---






