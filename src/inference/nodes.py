import os
import joblib

import pandas as pd
import numpy as np

from gensim.parsing.porter import PorterStemmer

from src.utils.feature_engineering import (
    process_text,
    feature_engineer,
)
from src.utils.bayes_opt_helpers import FEATURES


def apply_inference(
    input_path: str,
    output_path: str,
) -> None:
    """Some text...
    """
    # Import vectorizer and model
    vectorizer, model = joblib.load(f'{input_path}/training_output.pkl')

    # Read data
    inf_df = pd.read_csv(os.path.join(output_path, 'test_dataset.csv'))

    # Make copy
    X = inf_df.copy()

    # Apply feature engineering
    # Feature engineering
    X = feature_engineer(X)

    # Process text
    X['item_processed'] = X['item'].map(process_text)

    stemmer = PorterStemmer()
    X["item_processed"] = stemmer.stem_documents(X.item_processed.values)

    # Apply vectorizer
    X_tfidf = vectorizer.transform(X.item_processed.values).todense()
    X_combined = np.concatenate([X_tfidf, X[FEATURES].values.astype(float)], axis=1)

    # Apply model
    y_pred = model.predict_proba(X_combined)

    y_pred_df = pd.DataFrame(
        y_pred,
        columns=[
            'apparel_accessories',
            'home_garden_furniture',
            'other',
        ]
    )

    # Merge predictions to inference data
    res_df = pd.concat([inf_df, y_pred_df], axis=1)

    # Save output
    res_df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
