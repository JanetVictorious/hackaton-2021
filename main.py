import argparse
import os
import sys

from src.preprocessing.nodes import preprocess_data
from src.apply_split.nodes import split_data
from src.hpo_tuning.nodes import optimize_params


def _parse_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required arguments')

    #
    # Preprocess data
    #

    required.add_argument(
        "--pp_input_path",
        help="Preprocess input path.",
        dest='pp_input_path',
    )

    required.add_argument(
        "--pp_output_path",
        help="Preprocess output path.",
        dest='pp_output_path',
    )

    required.add_argument(
        "--params_path",
        help="Parameters path.",
        dest='pp_params_path',
        default='./config'
    )

    #
    # Split
    #

    required.add_argument(
        "--split_output_path",
        help="Split output path.",
        dest='split_output_path',
    )

    #
    # HPO tuning
    #

    required.add_argument(
        "--hpo_output_path",
        help="Preprocess input path.",
        dest='hpo_output_path',
    )

    required.add_argument(
        "--hpo_space",
        help="HPO space bounds.",
        dest='hpo_space',
    )

    #
    # Model training
    #

    #
    # Calibration
    #

    # drop unknown arguments
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    PARSER = _parse_args()

    preprocess_data(
        input_path=PARSER.pp_input_path,
        output_path=PARSER.pp_output_path,
        params_path=PARSER.params_path,
    )

    split_data(
        input_path=PARSER.pp_output_path,
        output_path=PARSER.split_output_path,
    )

    optimize_params(
        input_path=PARSER.split_output_path,
        output_path=PARSER.hpo_output_path,
        params_path=PARSER.params_path,
        hpo_space_name=PARSER.hpo_space,
    )
