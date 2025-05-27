"""
Demo module for BGAN.

This file is part of the BGAN framework for tabular data synthesis and imputation.

The demo data loading logic in this file is inspired by and partially adapted from the CTGAN codebase:
https://github.com/sdv-dev/CTGAN (MIT License).
The use of the census demo dataset and loading approach is based on the CTGAN demo.py implementation,
with modifications for BGAN.

MIT License applies to portions derived from CTGAN:
https://github.com/sdv-dev/CTGAN/blob/master/LICENSE

-------------------------------------------------------------------------------
"""

import pandas as pd

DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'


def load_demo():
    """
    Load the demo census dataset.

    Returns:
        pd.DataFrame: The census demo dataset loaded from the CTGAN demo URL.
    """
    return pd.read_csv(DEMO_URL, compression='gzip')
