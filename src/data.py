import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DATA_FILE, RANDOM_STATE, TARGET, TEST_SIZE


def load_data(path=DATA_FILE):
    return pd.read_csv(path)


def split_data(housing):
    """Stratified split on income category so the test set is representative."""
    income_cat = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_index, test_index = next(split.split(housing, income_cat))
    return housing.iloc[train_index], housing.iloc[test_index]


def split_features_target(df):
    return df.drop(TARGET, axis=1), df[TARGET]
