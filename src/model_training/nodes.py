import os
import json
import joblib

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.bayes_opt_helpers import FEATURES


def lr_model(
    input_path: str,
    output_path: str,
    params_path: str,
):
    """Some text...
    """
    # Read data
    train_df = pd.read_csv(os.path.join(input_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(input_path, 'val_data.csv'))

    # Set target name
    target_name = 'category'
    print(f'Target name: {target_name}')

    # Separate features and target
    x_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    x_val, y_val = val_df.drop(columns=[target_name]), val_df[target_name]

    X = pd.concat([x_train, x_val])
    y = pd.concat([y_train, y_val])

    # Import optimal hyper-parameters
    hpo_path = os.path.join(params_path, 'lr_hyperparameters.json')
    with open(hpo_path, 'r') as infile:
        hpo_params = json.load(infile)

    print(f'Optimal parameters: {hpo_params}')

    print('Create vectorizer...')
    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_df=hpo_params['max_df'],
        min_df=hpo_params['min_df'],
        strip_accents='ascii',
        max_features=10000,
        ngram_range=(1, hpo_params['ngrams']))

    X_tfidf = tfidf_vectorizer.fit_transform(X.item_processed.values)
    X_tfidf = X_tfidf.todense()

    # X_tfidf_val = tfidf_vectorizer.transform(x_val.item_processed.values).todense()

    # x_val = np.concatenate([X_tfidf_val, x_val[FEATURES].values.astype(float)], axis=1)
    X_combined = np.concatenate([X_tfidf, X[FEATURES].values.astype(float)], axis=1)

    print('Train model...')

    # Create model object
    lr = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        penalty='l2',
        C=hpo_params['alpha_inv'],
    )

    # Train data
    lr.fit(X_combined, y)

    print('Model trained!')

    # Save transformer and fitted model
    joblib.dump(
        (tfidf_vectorizer, lr),
        os.path.join(output_path, 'training_output.pkl')
    )
