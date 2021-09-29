import os
from posixpath import split

import pandas as pd

# Base path
basepath = os.path.dirname(__file__)

# Paths to data
train_data_filepath = os.path.abspath(os.path.join(basepath, "..", f'data/01_raw/hackathon_cleaned_dataset_v1.csv'))
df2_path = os.path.abspath(os.path.join(basepath, "..", f'data/02_feature_engineering/data.csv'))
split_train_data_filepath = os.path.abspath(os.path.join(basepath, "..", f'data/03_split/train_data.csv'))
sm_data_filepath = os.path.abspath(os.path.join(basepath, "..", f'data/06_inference/submission.csv'))

# Read data
train_df = pd.read_csv(train_data_filepath)
df2 = pd.read_csv(df2_path)
t_df = pd.read_csv(split_train_data_filepath)
sm_df = pd.read_csv(sm_data_filepath)

train_df['category'] = train_df['category'].astype('category').cat.codes.astype('int64')
train_df['category'].value_counts()

train_df.head()
train_df.shape
train_df.groupby('category').count()

df2.head()

t_df

sm_df.head()
sm_df.drop(columns=['item']).sum(axis=1)
