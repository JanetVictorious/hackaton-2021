import os

import pandas as pd

# Base path
basepath = os.path.dirname(__file__)

# Paths to data
train_data_filepath = os.path.abspath(os.path.join(basepath, "..", f'data/training_data.csv'))

# Read data
train_df = pd.read_csv(train_data_filepath)