"""
Data loading utilities for BGAN.

This file is part of the BGAN framework for tabular data synthesis and imputation.

The data loading and metadata handling logic in this file is inspired by and partially
adapted from the CTGAN codebase:
https://github.com/sdv-dev/CTGAN (MIT License).
The CSV/TSV reading, discrete/continuous column handling, and metadata parsing
are based on the CTGAN data.py implementation, with modifications for BGAN.

MIT License applies to portions derived from CTGAN:
https://github.com/sdv-dev/CTGAN/blob/master/LICENSE

-------------------------------------------------------------------------------
"""

import json

import numpy as np
import pandas as pd


def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    """
    Read a CSV file and extract discrete columns.

    Args:
        csv_filename (str): Path to the CSV file.
        meta_filename (str, optional): Path to metadata JSON file.
        header (bool): Whether the CSV has a header row.
        discrete (str, optional): Comma-separated list of discrete column names or indices.

    Returns:
        tuple: (data as pd.DataFrame, list of discrete column names or indices)
    """
    data = pd.read_csv(csv_filename, header='infer' if header else None)

    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name'] for column in metadata['columns'] if column['type'] != 'continuous'
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns


def read_tsv(data_filename, meta_filename):
    """
    Read a TSV file and extract discrete/continuous columns and metadata.

    Args:
        data_filename (str): Path to the TSV data file.
        meta_filename (str): Path to the metadata file.

    Returns:
        tuple: (data as np.ndarray, list of discrete column indices)
    """
    with open(meta_filename) as f:
        column_info = f.readlines()

    column_info_raw = [x.replace('{', ' ').replace('}', ' ').split() for x in column_info]

    discrete = []
    continuous = []
    column_info = []

    for idx, item in enumerate(column_info_raw):
        if item[0] == 'C':
            continuous.append(idx)
            column_info.append((float(item[1]), float(item[2])))
        else:
            assert item[0] == 'D'
            discrete.append(idx)
            column_info.append(item[1:])

    meta = {
        'continuous_columns': continuous,
        'discrete_columns': discrete,
        'column_info': column_info,
    }

    with open(data_filename) as f:
        lines = f.readlines()

    data = []
    for row in lines:
        row_raw = row.split()
        row = []
        for idx, col in enumerate(row_raw):
            if idx in continuous:
                row.append(col)
            else:
                assert idx in discrete
                row.append(column_info[idx].index(col))

        data.append(row)

    return np.asarray(data, dtype='float32'), meta['discrete_columns']


def write_tsv(data, meta, output_filename):
    """
    Write data and metadata to a TSV file.

    Args:
        data (np.ndarray): Data to write.
        meta (dict): Metadata dictionary with 'continuous_columns', 'discrete_columns', and 'column_info'.
        output_filename (str): Path to output TSV file.
    """
    with open(output_filename, 'w') as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)

            print(file=f)
