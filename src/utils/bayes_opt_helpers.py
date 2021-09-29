import pandas as pd
from typing import Dict, List

from sklearn.metrics import log_loss

from xgboost import XGBClassifier


def clf_log_loss(
    model_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    **kwargs
) -> float:
    """Some text...
    """
    # Create model object
    model = XGBClassifier(
        **model_params,
        nthread=-1,
        seed=42,
        verbosity=0
    )

    # Fit model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='logloss',
    )

    # Predictions on test data
    y_pred = model.predict(X_val)

    # Mean-absolute-error from predictions
    score = log_loss(y_val, y_pred)

    return float(-mae)
