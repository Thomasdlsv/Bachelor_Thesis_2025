from setuptools import setup, find_packages

setup(
    name="bgan",
    version="0.1.0",
    packages=find_packages(include=['bgan', 'bgan.*', 'ResearchQuestions', 'ResearchQuestions.*', 'bayesian_network', 'bayesian_network.*']),      
    python_requires=">=3.6",
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.3",
        "rdt>=1.15.0",
        "torch>=2.6.0",
        "tqdm>=4.67.1",
    ]
)