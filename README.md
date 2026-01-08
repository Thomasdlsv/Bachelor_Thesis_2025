# Synthetic Data Generation & Imputation  
Paper Title: Uncertainty-Aware Data Imputation Using Bayesian Network Guided Bayesian GANs

This repository contains code for benchmarking and evaluating synthetic data generation (SDG) and imputation methods, including BGAN, BN-AUG-SDG, CTGAN, and others. 

## Acknowledgments & Attribution

This implementation is built upon and inspired by the **CTGAN** project (https://github.com/sdv-dev/CTGAN), which introduced the conditional GAN approach for tabular data synthesis. Significant portions of the architecture, particularly the PAC discriminator, conditional sampling mechanisms, and data transformation pipeline, are adapted from CTGAN's MIT-licensed codebase.

**References:**
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling Tabular data using Conditional GAN*. NeurIPS 2019. https://arxiv.org/abs/1907.00503
- CTGAN Repository: https://github.com/sdv-dev/CTGAN (MIT License)

The BGAN and BN-AUG-SDG implementations extend CTGAN with Bayesian uncertainty estimation, KL regularization, and optional Bayesian Network-guided synthesis. For detailed licensing information, see the docstrings in `/bgan/synthesizers/bgan.py` and `/bgan/synthesizers/tvae.py`.

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
  
  Create an environment:
    
    python -m venv venv
    
  Activate the environment:
    
    .\venv\Scripts\Activate.ps1
  
  Make sure pip is up-to-date:
    
    pip install --upgrade pip
    
  Download the requirements for this project:
    
    pip install -r requirements.txt
    

---
  
## 3. Running Main Imputation Experiments

  **Primary experiment script (main results in report):**

  python tests/main.py
  
  This script performs the core imputation experiments from the report:
  - **Datasets tested:** Hepatitis, Heart Disease, Cancer (healthcare datasets as per report focus)
  - **Missing rates:** 10%, 20%, 30% Missing Completely At Random (MCAR)
  - **Imputation methods compared:**
    - BGAIN (Bayesian GAN-based imputation)
    - BN_AUG_Imputer (Bayesian Network-Augmented imputation - proposed method)
    - MICE (Multiple Imputation by Chained Equations)
    - KNN Imputer (k-Nearest Neighbors)
    - MissForest (Random Forest-based iterative imputation)
    - Mean/Mode (baseline statistical imputation)
  - **Evaluation metrics:**
    - Imputation quality (MAE, RMSE, MAPE)
    - Calibration metrics (reliability of uncertainty estimates)
    - Downstream task impact (classification accuracy on imputed data)
  - **Output location:** `outputs/imputation_data/`
  - **Number of repeats:** 5 repetitions per configuration (configurable at top of script)

---

## 4. Running Ablation & Analysis Studies

  **Loss Ablation Study:**

  ```bash
  python tests/loss_ablation_study.py
  ```

  Tests the impact of different loss components (uncertainty loss, KL divergence) on synthetic data quality:
  - Compares configurations: KL+CL (full), KL only, CL only, Adversarial only
  - Metrics: MMD, Diversity (Silhouette Score), Indistinguishability
  - Dataset: Cancer Dataset
  - Output location: `outputs/loss_ablation_data/`

  **Uncertainty Quantification Analysis:**

  ```bash
  python tests/uncertainty_analysis.py
  ```

  Analyzes uncertainty estimation quality between BGAIN and BN_AUG_Imputer:
  - Computes per-cell standard deviations across multiple stochastic imputations
  - Performs paired statistical tests (Wilcoxon, Cohen's d)
  - Visualizes uncertainty calibration
  - Output location: `outputs/imputation_data/uncertainty_per_cell_stds.csv`

  **Synthetic Data Quality Comparison:**

  ```bash
  python tests/synthetic_data_comparison.py
  ```

  Compares synthetic data generators on distribution fidelity:
  - Methods: BN-BGAN (proposed), BGAN, CTGAN, TVAE
  - Datasets: Healthcare datasets from report
  - Metrics: MMD, Diversity, Classification accuracy (indistinguishability)
  - Output location: `outputs/synthetic_data_comparison/`

---

## 5. Advanced: SDG Development Tests

  **Synthetic Data Generator Development (optional):**

  ```bash
  python -m tests.sdg_tests.main
  ```

  For development and visualization of synthetic data generation pipeline:
  - Trains BGAN, BN-AUG-SDG, and CTGAN models
  - Visualizes DAG structure learned by Bayesian Network
  - Compares feature distributions between real and synthetic data
  - Note: Results from this are supplementary; main report uses `synthetic_data_comparison.py`

  **Bayesian Network Structure Uncertainty:**

  ```bash
  python -m tests.sdg_tests.uncertainty_bn_test
  ```

  Analyzes uncertainty in the learned Bayesian Network structures
  - Examines robustness of structure learning
  - Supplementary analysis for report

## 6. Configurations

  **Imputation Experiment Parameters** (in `tests/main.py`):
  - `N_REPEATS`: Number of repetitions per configuration (default: 5)
  - `MISSING_RATES`: Missing data rates to test (default: [0.1, 0.2, 0.3])
  - `RANDOM_SEED`: Base seed for reproducibility (default: 42)
  - `EPOCHS`: Training epochs for neural methods (default: 50)
  - `BATCH_SIZE`: Batch size for training (default: 100)
  - `LEARNING_RATE`: Learning rate for optimizers (default: 2e-4)

  **Datasets** (in `tests/main.py`):
  - Configure which datasets to run in the `DATASETS` list
  - Uncomment additional datasets (e.g., Diabetes) as needed
  - Each dataset requires: `name`, `path`, and `target` column specification

  **Loss Ablation Study Parameters** (in `tests/loss_ablation_study.py`):
  - Customize loss configurations, datasets, and evaluation metrics
  - Adjust training epochs, batch sizes, and hyperparameters

  **Uncertainty Analysis Parameters** (in `tests/uncertainty_analysis.py`):
  - `n_imputations`: Number of stochastic imputations per model (default: 30)
  - `epochs`: Training epochs for imputation models (default: 50)
  - `test_size`: Train/test split ratio (default: 0.2)

---

## 7. Customization and Advanced Usage

  **Adding New Imputation Methods:**
  - Modify `create_imputation_methods()` in `tests/main.py`
  - Implement interface with `impute_all_missing(X)` method

  **Adding New Datasets:**
  - Add entry to `DATASETS` list with name, path, and target column
  - Supports: ARFF format and CSV files (from `new_datasets/`)
  - Ensure proper handling of continuous and categorical features

  **Controlling Model Components:**
  - **BN-AUG-SDG Influence:** Modify `bn_influence` parameter in SDG scripts
  - **Batch Normalization:** Toggle `batch_norm` in `BN_AUG_SDG` constructor
  - **Uncertainty Loss:** Toggle `use_uncertainty_loss` in BGAN/Imputer constructors
  - **KL Regularization:** Toggle `use_kl_loss` in BGAN/Imputer constructors

  **Visualization & Results:**
  - All results automatically saved to `outputs/` subdirectories
  - CSV outputs can be further analyzed with external tools
  - Plots are generated using matplotlib/seaborn

---

## 8. Datasets

  **Primary Datasets** (used in report experiments):
  - **Hepatitis Dataset:** Mixed-type healthcare data on hepatitis patients
    Location: `new_datasets/mixed_data_hepatisis_dataset`
  - **Heart Disease Dataset:** Baseline heart disease prediction dataset
    Location: `new_datasets/baseline_heart_disease_dataset`
  - **Cancer Dataset:** Classification dataset for cancer diagnosis
    Location: `datasets/Cancer_Dataset.arff`

  **Optional Datasets** (available but not in main report):
  - **Diabetes Dataset:** Large diabetes prediction dataset
    Location: `new_datasets/large_diabetes_dataset`
  - **Fetal Dataset:** Fetal health monitoring data
    Location: `datasets/Fetal_Dataset.arff`

  **Dataset Sources:**
  - All datasets are open-source from OpenML or publicly available healthcare repositories
  - Focus is on healthcare/medical classification tasks for domain relevance
  - Datasets are stored in ARFF or CSV format

  **Adding New Datasets:**
  - Place ARFF or CSV files in `datasets/` or `new_datasets/` folders
  - Update the `DATASETS` configuration in relevant test scripts
  - Ensure proper specification of `target` column for classification tasks

  **Demonstration Dataset:**
  - U.S. Census dataset available at: http://ctgan-demo.s3.amazonaws.com/census.csv.gz
  - Useful for testing pipelines and BN structure visualization
  - Not used in main experimental evaluations from the report

---

## 9. Troubleshooting

  **General Issues:**
  - If you encounter missing package errors, ensure your environment is activated and run `pip install -r requirements.txt` again.
  - For CUDA/GPU issues, the code will automatically fall back to CPU. Set `cuda=False` in model constructors if needed.

  **Experiment Runtime:**
  - Large datasets or high epoch counts will increase runtime significantly
  - To test setup quickly, reduce `N_REPEATS`, `EPOCHS`, or use smaller `MISSING_RATES`
  - Typical runtime for main imputation experiment: 30-60 minutes (5 repeats, 3 datasets, 50 epochs)

  **Memory Issues:**
  - Reduce `BATCH_SIZE` if encountering out-of-memory errors
  - Use fewer datasets or lower missing rates for testing on limited hardware

  **Output & Results:**
  - Results are automatically saved to `outputs/` subdirectories as CSV files
  - If output folder doesn't exist, it will be created automatically
  - Check console output and CSV files for detailed results and metrics

  **Reproducibility:**
  - All experiments use fixed random seeds (configurable)
  - Set `RANDOM_SEED` at top of scripts for consistent results
  - Use same package versions from `requirements.txt` for full reproducibility

  **Model-Specific Issues:**
  - **BN_AUG_Imputer:** Requires sufficient data for Bayesian Network structure learning
  - **BGAIN:** May require more epochs on complex datasets
  - **CTGAN:** If not installed, install separately: `pip install ctgan`






