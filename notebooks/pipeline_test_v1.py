import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from ruamel.yaml import YAML

from bayes_opt import BayesianOptimization

pd.set_option('max.rows', 100)
pd.set_option('max.columns', 100)

os.chdir('..')

import src
sys.modules['src'] = src

from src.utils.feature_engineering import (
    process_text,
    feature_engineer,
)
from src.utils.bayes_opt_helpers import xgb_log_loss, lr_log_loss

# Base path
basepath = os.path.dirname(__file__)


# +-------------------------+
# |   Feature engineering   |
# +-------------------------+

# Paths to data
raw_df_path = os.path.abspath(os.path.join(basepath, "..", 'data/01_raw/hackathon_cleaned_dataset_v1.csv'))

# Read data
raw_df = pd.read_csv(raw_df_path)
raw_df.head()

# Remove misc columns
raw_df = raw_df.drop(columns=['prediction', 'correct', 'prediction_proba'])
raw_df.shape
raw_df.head()

# Feature eng
df = feature_engineer(raw_df)
df.head()

# Process text
df['item_processed'] = df['item'].map(process_text)
df.head()

df['category'] = df['category'].astype('category').cat.codes.astype('int64')

df.head()
df['category'].value_counts()


# +----------------+
# |   Split data   |
# +----------------+

# Split data
df_train, df_val = train_test_split(df, test_size=0.20, random_state=42)
df_train.shape
df_val.shape


# +------------------------+
# |   XGBoost HPO tuning   |
# +------------------------+

yaml = YAML()

hpo_path = os.path.abspath(os.path.join(basepath, "..", 'config/hpo_params.yml'))

with open(hpo_path, 'r') as infile:
    hpo_space = yaml.load(infile)

hpo_space
hpo_space['colsample_bytree']

target_name = 'category'
col_to_drop = ['item', 'item_processed', 'category']

df_train.head()

x_train, y_train = df_train.drop(columns=col_to_drop), df_train[target_name]
x_val, y_val = df_val.drop(columns=col_to_drop), df_val[target_name]


# Hyper-parameter tuning
def log_loss_score(colsample_bytree,
                   colsample_bylevel,
                   colsample_bynode,
                   learning_rate,
                   max_depth,
                   n_estimators,
                   subsample,
                   min_child_weight,
                   reg_alpha,
                   reg_lambda,
                   gamma):
    """Wrapper of XGBClassifier logloss"""
    model_params = {
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators),
        'subsample': subsample,
        'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'gamma': gamma,
    }
    return xgb_log_loss(
        model_params=model_params,
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
    )


optimizer = BayesianOptimization(
    f=log_loss_score,
    pbounds=hpo_space,
    random_state=42,
    verbose=2,
)

optimizer.maximize(n_iter=10)

optimizer.max
optimizer.max['params']


# +------------------------------------+
# |   Logistic regression HPO tuning   |
# +------------------------------------+
