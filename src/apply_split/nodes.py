import os

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

def split_data(
    input_path: str,
    output_path: str,
) -> None:
    """Some text...
    """
    # Data path
    data_path = os.path.join(input_path, 'data.feather')

    # Read data
    df = pd.read_feather(data_path)

    # Split data
    df_train, df_val = train_test_split(
        df, test_size=0.33, random_state=42)

    # Export datasets
    df_train.to_feather(os.path.join(output_path, 'train_data.feather'))
    df_val.to_feather(os.path.join(output_path, 'val_data.feather'))
