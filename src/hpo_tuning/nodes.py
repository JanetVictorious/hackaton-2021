import json
import os

import pandas as pd
from ruamel.yaml import YAML

from bayes_opt import BayesianOptimization

from src.utils.bayes_opt_helpers import xgb_log_loss, lr_log_loss


def xgb_optimize_params(
    input_path: str,
    output_path: str,
    params_path: str = None,
    hpo_space_name: str = None,
) -> None:
    """Some text...
    """
    # Create yaml object
    yaml = YAML()

    # HPO space filepath
    hpo_path = os.path.join(params_path, hpo_space_name)

    # Open hpo_space.yml as hpo_space
    with open(hpo_path, 'r') as infile:
        hpo_space = dict(yaml.load(infile))

    print((f'HPO space: {hpo_space}'))

    # Read data
    train_df = pd.read_csv(os.path.join(input_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(input_path, 'val_data.csv'))

    print(f'Shape of training data: {train_df.shape}')
    print(f'Shape of validation data: {val_df.shape}')

    # Set target name
    target_name = 'category'
    print(f'Target name: {target_name}')

    # Separate features and target
    x_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    x_val, y_val = val_df.drop(columns=[target_name]), val_df[target_name]

    print(f'X_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_val shape: {x_val.shape}')
    print(f'y_val shape: {y_val.shape}')

    print('Creating hpo pipeline...')

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
                       gamma,
                       max_df,
                       min_df,
                       ngrams):
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
            max_df=float(max_df),
            min_df=int(min_df),
            ngrams=int(ngrams),
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

    print(f'Final result: {optimizer.max}')

    # Extract optimal parameters from Bayesian optimization
    opt_params = optimizer.max['params']

    # Cast int parameters from float to int
    opt_params['max_depth'] = int(round(opt_params['max_depth']))
    opt_params['n_estimators'] = int(round(opt_params['n_estimators']))
    opt_params['min_df'] = int(round(opt_params['min_df']))
    opt_params['ngrams'] = int(round(opt_params['ngrams']))

    print(f'Optimal parameters: {opt_params}')

    # Export optimal hyper-parameters
    with open(os.path.join(output_path, 'xgb_hyperparameters.json'), 'w') as outfile:
        json.dump(opt_params, outfile)


def optimize_lr_params(
    input_path: str,
    output_path: str,
    params_path: str = None,
    hpo_space_name: str = None,
) -> None:
    """Some text...
    """
    # Create yaml object
    yaml = YAML()

    # HPO space filepath
    hpo_path = os.path.join(params_path, hpo_space_name)

    # Open hpo_space.yml as hpo_space
    with open(hpo_path, 'r') as infile:
        hpo_space = dict(yaml.load(infile))

    print((f'HPO space: {hpo_space}'))

    # Read data
    train_df = pd.read_csv(os.path.join(input_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(input_path, 'val_data.csv'))

    print(f'Shape of training data: {train_df.shape}')
    print(f'Shape of validation data: {val_df.shape}')

    # Set target name
    target_name = 'category'
    print(f'Target name: {target_name}')

    # Separate features and target
    x_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    x_val, y_val = val_df.drop(columns=[target_name]), val_df[target_name]

    print(f'X_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_val shape: {x_val.shape}')
    print(f'y_val shape: {y_val.shape}')

    print('Creating hpo pipeline...')

    # Hyper-parameter tuning
    def lr_log_loss_score(max_df,
                          min_df,
                          ngrams,
                          alpha_inv):
        """Wrapper of logloss"""
        return lr_log_loss(
            max_df=float(max_df),
            min_df=int(min_df),
            ngrams=int(ngrams),
            alpha_inv=float(alpha_inv),
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
        )

    optimizer = BayesianOptimization(
        f=lr_log_loss_score,
        pbounds=hpo_space,
        random_state=42,
        verbose=2,
    )
    optimizer.maximize(n_iter=10)

    print(f'Final result: {optimizer.max}')

    # Extract optimal parameters from Bayesian optimization
    opt_params = optimizer.max['params']

    # Cast int parameters from float to int
    opt_params['min_df'] = int(round(opt_params['min_df']))
    opt_params['ngrams'] = int(round(opt_params['ngrams']))

    print(f'Optimal parameters: {opt_params}')

    # Export optimal hyper-parameters
    with open(os.path.join(output_path, 'lr_hyperparameters.json'), 'w') as outfile:
        json.dump(opt_params, outfile)
