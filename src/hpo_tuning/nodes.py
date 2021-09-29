import json
import os

import pandas as pd
from ruamel.yaml import YAML

from bayes_opt import BayesianOptimization

from src.utils.bayes_opt_helpers import clf_log_loss


def optimize_params(
    data_path: str,
    output_path: str,
    params_path: str = None,
    hpo_space_name: str = None,
) -> None:
    """Apply Bayesian Optimization to regressor parameters.

    :param str output_dir:
        Output save path.
    :param str metadata_file:
        Metadata file path.
    :param str train_data_feather_path:
        Training data feather path.
    :param str test_data_feather_path:
        Test data feather path.
    :param str hyperparameter_fname:
        NOT USED...
    :param str experiment_path:
        Experiment path.
    :param str experiment_name:
        Experiment filename.
    :param str hpo_space_name:
        HPO space filename.

    :returns:
        Json with optimized hyper-parameters.
        :param test_data_feather_path:
        :param output_save_path:
        :param metadata_file:
        :param train_data_feather_path:
        :param hpo_space_name:
        :param experiment_name:
        :param experiment_path:
        :param feature_selection_path:
        :param val_data_feather_path:
    """
    # Create yaml object
    yaml = YAML()

    # HPO space filepath
    hpo_path = os.path.join(params_path, hpo_space_name)

    # Open hpo_space.yml as hpo_space
    with open(hpo_path, 'r') as infile:
        hpo_space = yaml.load(infile)

    print((f'HPO space: {hpo_space}'))

    # Read data
    train_df = pd.read_feather(f'{feature_selection_dir}/pp_train_data.feather')
    val_df = pd.read_feather(f'{feature_selection_dir}/pp_val_data.feather')

    print(f'Shape of training data: {train_df.shape}')
    print(f'Shape of validation data: {val_df.shape}')

    # Set target name
    target_name = 'target'
    print(f'Target name: {target_name}')

    # Separate features and target
    x_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    x_val, y_val = val_df.drop(columns=[target_name]), val_df[target_name]

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
        return clf_log_loss(
            model_params=model_params,
            data=x_train,
            targets=y_train,
            data_val=x_val,
            targets_val=y_val,
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

    print(f'Optimal parameters: {opt_params}')

    # Export optimal hyper-parameters
    with open(os.path.join(output_path, f'hyperparameters.json'), 'w') as outfile:
        json.dump(opt_params, outfile)
