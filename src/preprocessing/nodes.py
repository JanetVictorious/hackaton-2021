import os

import pandas as pd

import yaml
import joblib

from src.utils.params_metadata import HackatonMetaData
from src.utils.preprocessing import (
    process_text,
)


def preprocess_data(
    input_path: str,
    output_path: str,
    params_path: str,
) -> None:
    """Some text...
    """
    # Data path
    data_path = os.path.join(input_path, 'training_dataset.csv')

    # # Path to metadata
    # metadata_path = os.path.join(params_path, 'metadata.yml')

    # # Open metadata
    # with open(metadata_path, 'r') as infile:
    #     raw_metadata = yaml.load(infile)

    # # Create metadata object
    # metadata = HackatonMetaData(raw_metadata)

    # Read data
    df_raw = pd.read_csv(data_path)

    # Copy data
    df = df_raw.copy()

    # Process text
    df['item_processed'] = df['item'].map(process_text)

    # # Create dtypes dict
    # dtypes_dict = metadata.get_dtypes_dict()
    
    # # Data type conversion
    # df = df.astype(dtypes_dict)

    # Export dataset
    df.to_feather(os.path.join(output_path, 'data.feather'))

    # # Export metadata object
    # joblib.dump(metadata, os.path.join(output_path, 'metadata.pkl'))
