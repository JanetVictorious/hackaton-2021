import os

import pandas as pd

from gensim.parsing.porter import PorterStemmer

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

    # stemmer = PorterStemmer()
    # df["item_processed"] = stemmer.stem_documents(df.item_processed.values)

    print('Feature engineering done!')

    # Create target feature
    df['category'] = df['category'].astype('category').cat.codes.astype('int64')

    df = df.reset_index(drop=True)

    # Export dataset
    print('Export data...')
    df.to_csv(os.path.join(output_path, 'data.csv'), index=False)
