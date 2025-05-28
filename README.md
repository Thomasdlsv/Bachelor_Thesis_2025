# Synthetic Data Generation & Imputation  
**Maastricht University**  
**Bachelor of Data Science and Artificial Intelligence**  
**Author:** Thomás de los Santos Verrijp

This repository contains code for benchmarking and evaluating synthetic data generation (SDG) and imputation methods, including BGAN, BN-AUG-SDG, CTGAN, and others.

## Table of Contents
1. [Environment Setup](#environment-setup)  
2. [Running SDG Evaluations](#running-sdg-evaluations)  
3. [Running Imputation Evaluations](#running-imputation-evaluations)  
4. [Configurations](#configurations)  
5. [Customization and Advanced Usage](#customization-and-advanced-usage)  
6. [Troubleshooting](#troubleshooting)

---

## 1. Environment Setup

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
  
## 2. Running SDG Evaluations

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

---

## 3. Running Imputation Evaluations

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
   
## 4. Configurations
    - **Experiment parameters** (number of repetitions, missing rates, random seed) can be set at the top of `tests/imputation_tests/main.py`.
    - **Evaluation pipeline** missingness rates (10%, 20%,...), and missingness patterns (MCAR, MAR, MNAR) are handled in `tests/imputation_tests/configurations.py`.
    - You can add or remove imputation methods and adjust missingness patterns as needed.

---
  
## 5. Customization and Advanced Usage
  - All evaluation and visualization logic is modularized in the `SDGVisualizer` class (`tests/sdg_tests/sdg_visualizer.py`).
  - You can run only specific visualizations or metrics by calling the relevant methods from this class.
  - For hyperparameter search, adjust the parameter grids in the main scripts.
  - For new datasets, update the data loading section in the relevant script.

---

## 6. Troubleshooting
  - If you encounter missing package errors, ensure your environment is activated and run `pip install -r requirements.txt` again.
  - For large datasets or long runs, reduce the number of epochs or sample sizes in the configuration.
  - When running a hyperparameter search, ensure to output the text of only the parameters searched, otherwise can lead to errors.

---





