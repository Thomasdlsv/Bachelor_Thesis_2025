from setuptools import setup, find_packages

setup(
    name="bachelor_thesis_2025",
    version="0.1.0",
    packages=[
        "bayesian_network",
        "bayesian_network.*",
        "bgan",
        "bgan.*",
        "imputation",
        "imputation.*",
        "ResearchQuestions",
        "ResearchQuestions.*",
        "bgain",
        "bgain.*"
    ],
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.2",
        "scipy>=1.7.0",
        "ctgan>=0.7.0",
        "pgmpy>=0.1.20",
        "networkx>=2.6.0"
    ],
    python_requires='>=3.8',
    author="Thomas",
    description="Synthetic Data Generation Research",
)