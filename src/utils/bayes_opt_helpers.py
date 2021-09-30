import pandas as pd
import numpy as np
from typing import Dict, List

from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier

FEATURES = ['count_sizes', 'special_char_count', 'number_count', 'fraction_upper_words', 'fraction_upper_start_words',
            'has_volume', 'has_weight', 'has_uk_size', 'has_area', 'has_color', 'spaces', 'has_cm', 'has_m', 'has_xn',
            'has_french_combinations']


def xgb_log_loss(
    model_params: Dict,
    max_df: float,
    min_df: int,
    ngrams: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    **kwargs
) -> float:
    """Some text...
    """
    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df,
                                       min_df=min_df,
                                       strip_accents='ascii',
                                       max_features=10000,
                                       ngram_range=(1, ngrams))

    X_tfidf_train = tfidf_vectorizer.fit_transform(X_train.item_processed.values)
    X_tfidf_train = X_tfidf_train.todense()

    X_tfidf_val = tfidf_vectorizer.transform(X_val.item_processed.values).todense()

    x_train = np.concatenate([X_tfidf_train, X_train[FEATURES].values.astype(float)], axis=1)
    x_val = np.concatenate([X_tfidf_val, X_val[FEATURES].values.astype(float)], axis=1)

    # Create model object
    model = XGBClassifier(
        **model_params,
        nthread=-1,
        seed=42,
        verbosity=0,
    )

    # Fit model
    model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='mlogloss',
    )

    # Predictions on test data
    y_pred = model.predict_proba(x_val)

    # print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
    # print(f'F1 score: {f1_score(y_val, y_pred)}')

    # Mean-absolute-error from predictions
    score = log_loss(y_val, y_pred)

    return float(-score)


def lr_log_loss(
    max_df,
    min_df,
    ngrams,
    alpha_inv,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Some text...
    """
    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, strip_accents='ascii',
                                       max_features=10000, ngram_range=(1, ngrams))

    X_tfidf_train = tfidf_vectorizer.fit_transform(X_train.item_processed.values)
    X_tfidf_train = X_tfidf_train.todense()

    X_tfidf_val = tfidf_vectorizer.transform(X_val.item_processed.values).todense()

    x_val = np.concatenate([X_tfidf_val, X_val[FEATURES].values.astype(float)], axis=1)
    x_train = np.concatenate([X_tfidf_train, X_train[FEATURES].values.astype(float)], axis=1)

    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, penalty='l2', C=alpha_inv)
    lr.fit(x_train, y_train)

    y_prob = lr.predict_proba(x_val)
    y_pred = lr.predict(x_val)

    print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
    print(f"F1 score: {f1_score(y_val, y_pred, average='micro')}")

    ll = log_loss(y_val, y_pred=y_prob)

    return float(-ll)
