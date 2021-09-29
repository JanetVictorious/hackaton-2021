import os

import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(
    input_path: str,
    output_path: str,
) -> None:
    """Some text...
    """
    # Data path
    data_path = os.path.join(input_path, 'data.csv')

    # Read data
    df = pd.read_csv(data_path)

    print('Splitting data...')

    # Split data
    df_train, df_val = train_test_split(
        df, test_size=0.20, random_state=42)

    print('Data splitted!')
    print(f'Shape of training data: {df_train.shape}')
    print(f'Shape of validation data: {df_val.shape}')

    # Export datasets
    df_train.to_csv(os.path.join(output_path, 'train_data.csv'), index=False)
    df_val.to_csv(os.path.join(output_path, 'val_data.csv'), index=False)
