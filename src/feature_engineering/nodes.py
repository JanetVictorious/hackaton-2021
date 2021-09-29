import os

import pandas as pd

import yaml
import joblib

from gensim.parsing.porter import PorterStemmer

from src.utils.params_metadata import HackatonMetaData
from src.utils.feature_engineering import (
    process_text,
    feature_engineer,
)


def feature_engineer_data(
    input_path: str,
    output_path: str,
    params_path: str,
) -> None:
    """Some text...
    """
    # Data path
    data_path = os.path.join(input_path, 'hackathon_cleaned_dataset_v1.csv')

    # # Path to metadata
    # metadata_path = os.path.join(params_path, 'metadata.yml')

    # # Open metadata
    # with open(metadata_path, 'r') as infile:
    #     raw_metadata = yaml.load(infile)

    # # Create metadata object
    # metadata = HackatonMetaData(raw_metadata)

    # Read data
    print('Read data...')
    df_raw = pd.read_csv(data_path)

    # Remove misc columns
    df_raw = df_raw.drop(columns=['prediction', 'correct', 'prediction_proba'])

    # Copy data
    df = df_raw.copy()
    print(f'Shape of data: {df.shape}')

    print('Start preprocessing...')

    # Feature engineering
    df = feature_engineer(df)

    # Process text
    df['item_processed'] = df['item'].map(process_text)

    stemmer = PorterStemmer()
    df["item_processed"] = stemmer.stem_documents(df.item_processed.values)

    print('Feature engineering done!')

    # # Create dtypes dict
    # dtypes_dict = metadata.get_dtypes_dict()

    # # Data type conversion
    # df = df.astype(dtypes_dict)

    # Create target feature
    df['category'] = df['category'].astype('category').cat.codes.astype('int64')

    df = df.reset_index(drop=True)

    # Export dataset
    print('Export data...')
    df.to_csv(os.path.join(output_path, 'data.csv'), index=False)

    # # Export metadata object
    # joblib.dump(metadata, os.path.join(output_path, 'metadata.pkl'))
