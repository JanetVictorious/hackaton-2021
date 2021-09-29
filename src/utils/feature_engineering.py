import pandas as pd
import numpy as np

import re

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
english_stopwords = stopwords.words('english')

color_names = [
    "pink", "red", "green", "mint", "black", "grey", "gray", "silver", "gold",
    "yellow", "purple", "blue", "white", "lime", "navy", "teal", "olive",
    "brown"
]


def process_text(text: str) -> str:
    """ Remove all number, special characters, single letters, etc."""
    text = text.replace('-', ' ')
    processed_words = []
    for w in text.split(' '):
        w = ''.join(c for c in w if c.isalpha())
        w = w.lower()
        if w not in english_stopwords and len(w) > 1:
            processed_words.append(w)
    return ' '.join(processed_words)


def has_color(item: str):
    item = item.lower()
    for color in color_names:
        if color in item:
            return True
    return False


# def feature_engineer(raw_df: pd.DataFrame) -> pd.DataFrame:
#     df = raw_df.copy()

#     df['number_count'] = df.item.map(lambda x: sum(c.isnumeric() for c in x))
#     df['fraction_upper_words'] = df.item.map(lambda x: np.mean([w.isupper() for w in x.split(' ') if len(w) > 1]))
#     df['fraction_upper_start_words'] = df.item.map(lambda x: np.mean([w[0].isupper() for w in x.split(' ') if len(w) > 1]))
#     df['special_char_count'] = df.item.map(lambda x: len([c for c in x if not c.isalnum()]))
#     df["has_volume"] = df.item.apply(lambda x: re.match('.*[0-9]+ml.*', x) is not None)
#     df["has_weight"] = df.item.apply(lambda x: re.match('.*[0-9]+k?g.*', x) is not None)
#     df["has_uk_size"] = df.item.apply(lambda x: re.match('.*UK [0-9]+.*', x) is not None)
#     df["has_area"] = df.item.apply(lambda x: re.match('.*\([0-9]+(cm)?x[0-9]+.*', x) is not None)
#     df["has_color"] = df.item.apply(has_color)
#     df['spaces'] = df['item'].apply(lambda x: len(x.split()))
#     sizes = {'XXL', 'XXXL', 'L', 'M', 'S', 'XXS', 'XXS', 'XXS'}
#     df['count_sizes'] = df.item.map(lambda x: sum(w in sizes for w in x.split(' ')))

#     return df.reset_index(drop=True)


def feature_engineer(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    sizes = {'XXL', 'XXXL', 'L', 'M', 'S', 'XXS', 'XXS', 'XXS'}
    df['count_sizes'] = df.item.map(lambda x: sum(w in sizes for w in x.split(' ')))

    df['number_count'] = df.item.map(lambda x: sum(c.isnumeric() for c in x))
    df['fraction_upper_words'] = df.item.map(lambda x: np.mean([w.isupper() for w in x.split(' ') if len(w) > 1]))
    df['fraction_upper_start_words'] = df.item.map(lambda x: np.mean([w[0].isupper() for w in x.split(' ') if len(w) > 1]))
    df['special_char_count'] = df.item.map(lambda x: len([c for c in x if not c.isalnum()]))

    df["has_volume"] = df.item.apply(lambda x: re.match('.*[0-9]+ml.*', x) is not None)
    df["has_weight"] = df.item.apply(lambda x: re.match('.*[0-9]+k?g.*', x) is not None)
    df["has_uk_size"] = df.item.apply(lambda x: re.match('.*UK [0-9]+.*', x) is not None)
    df["has_area"] = df.item.apply(lambda x: re.match('.*\([0-9]+(cm)?x[0-9]+.*', x) is not None)
    df["has_color"] = df.item.apply(has_color)

    df['spaces'] = df['item'].apply(lambda x: len(x.split()))

    df["has_cm"] = df.item.apply(lambda x: re.match('.*[0-9]+cm.*', x) is not None)
    df["has_m"] = df.item.apply(lambda x: re.match('.*[0-9]+m.*', x) is not None)
    df["has_xn"] = df.item.apply(lambda x: re.match('.*x[0-9]+.*', x) is not None)
    df["has_french_combinations"] = df.item.apply(lambda x: re.match('.*((eu)|(au )|(aux)).*', x.lower()) is not None)

    return df.reset_index(drop=True)


def check_features(d: pd.DataFrame, features: list):
    for feat in features:
        print(feat)
        print(d.groupby('category').apply(lambda df: df[feat].mean()))
        print()


# check_features(df, ['count_sizes', 'special_char_count', 'number_count', 'fraction_upper_words', 'fraction_upper_start_words',
#                     'has_volume', 'has_weight', 'has_uk_size', 'has_area', 'has_color', 'spaces'])
