import argparse

from src.feature_engineering.nodes import feature_engineer_data
from src.apply_split.nodes import split_data
from src.hpo_tuning.nodes import xgb_optimize_params, optimize_lr_params
from src.model_training.nodes import lr_model
from src.inference.nodes import apply_inference


def _parse_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required arguments')

    #
    # Feature engineering
    #

    required.add_argument(
        "--fe_input_path",
        help="Feature engineering input path.",
        dest='fe_input_path',
        default='./data/01_raw'
    )

    required.add_argument(
        "--fe_output_path",
        help="Feature engineering output path.",
        dest='fe_output_path',
        default='./data/02_feature_engineering'
    )

    required.add_argument(
        "--params_path",
        help="Parameters path.",
        dest='params_path',
        default='./config'
    )

    #
    # Split
    #

    required.add_argument(
        "--split_output_path",
        help="Split output path.",
        dest='split_output_path',
        default='./data/03_split',
    )

    #
    # HPO tuning
    #

    required.add_argument(
        "--hpo_output_path",
        help="Preprocess input path.",
        dest='hpo_output_path',
        default='./data/04_hpo_tuning'
    )

    required.add_argument(
        "--hpo_space",
        help="HPO space bounds.",
        dest='hpo_space',
        default='hpo_params.yml',
    )

    required.add_argument(
        "--lr_hpo_space",
        help="HPO space bounds.",
        dest='lr_hpo_space',
        default='lr_hpo_params.yml',
    )

    #
    # Model training
    #

    required.add_argument(
        "--lr_model_output_path",
        help="Preprocess input path.",
        dest='lr_model_output_path',
        default='./data/05_model_training',
    )

    #
    # Inference
    #

    required.add_argument(
        "--inf_output_path",
        help="Preprocess input path.",
        dest='inf_output_path',
        default='./data/06_inference',
    )

    # drop unknown arguments
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    PARSER = _parse_args()

    # #
    # # Feature engineering
    # #

    # feature_engineer_data(
    #     input_path=PARSER.fe_input_path,
    #     output_path=PARSER.fe_output_path,
    #     params_path=PARSER.params_path,
    # )

    # #
    # # Split
    # #

    # split_data(
    #     input_path=PARSER.fe_output_path,
    #     output_path=PARSER.split_output_path,
    # )

    #
    # HPO tuning
    #

    xgb_optimize_params(
        input_path=PARSER.split_output_path,
        output_path=PARSER.hpo_output_path,
        params_path=PARSER.params_path,
        hpo_space_name=PARSER.hpo_space,
    )

    # optimize_lr_params(
    #     input_path=PARSER.split_output_path,
    #     output_path=PARSER.hpo_output_path,
    #     params_path=PARSER.params_path,
    #     hpo_space_name=PARSER.lr_hpo_space,
    # )

    # #
    # # Model training
    # #

    # lr_model(
    #     input_path=PARSER.split_output_path,
    #     output_path=PARSER.lr_model_output_path,
    #     params_path=PARSER.hpo_output_path,
    # )

    # #
    # # Inference
    # #

    # apply_inference(
    #     input_path=PARSER.lr_model_output_path,
    #     output_path=PARSER.inf_output_path,
    # )
