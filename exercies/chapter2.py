import os
import unittest
from unittest import TestCase
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOUSING_PATH = os.path.join("../datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH) -> pd.DataFrame:
    return pd.read_csv(os.path.join(housing_path, "housing.csv"))


def split_train_test(data, test_ratio):
    data_size = len(data)
    shuffled_indices = np.random.permutation(data_size)
    test_set_size = int(data_size * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


class TestHouseAnalysis(TestCase):

    def setUp(self) -> None:
        self.data = load_housing_data()

    def test_split_set(self):
        train, test = split_train_test(self.data, 0.2)
        print("train set size {},test set size {}".format(len(train), len(test)))
        train, test = train_test_split(self.data, test_size=0.2)
        print("train set size {},test set size {}".format(len(train), len(test)))
        self.assertTrue(True)

    def test_data_info(self):
        data = self.data
        print("header is:")
        print(data.head(5))
        print("info is:")
        print(data.info())
        print("ocean_proximity value counts:")
        print(data['ocean_proximity'].value_counts())
        print("data's describe:")
        print(data.describe())
        data.hist(bins=50, figsize=(30, 15))
        plt.show()
        self.assertTrue(True)

    def test_split_data(self):
        train, test = split_train_test(self.data, test_ratio=0.2)
        print("train size is {}, and test size is {}".format(len(train), len(test)))

        self.data['income_cat'] = np.ceil(self.data['median_income'] / 1.5)
        print("income numbers before set 5:")
        print(self.data['income_cat'].value_counts())
        self.data['income_cat'].where(self.data['income_cat'] < 5, 5.0, inplace=True)

        self.split_analysis_set()

        print("income numbers:")
        print(self.data['income_cat'].value_counts())
        print("income ratio:")
        print(self.data['income_cat'].value_counts() / len(self.data))
        print("income describe:")
        print(self.data['income_cat'].describe())
        self.assertTrue(True)

    def test_plot_data(self):
        train_set, test_set = self.split_analysis_set()
        housing = train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing['population'] / 100,
                     label="population", c="median_house_value", cmap=plt.get_cmap('jet'), colorbar=True)
        plt.show()
        self.assertTrue(True)

    def split_analysis_set(self) -> (pd.DataFrame, pd.DataFrame):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        strat_train_set = None
        strat_test_set = None
        self.data['income_cat'] = np.ceil(self.data['median_income'] / 1.5)
        self.data['income_cat'].where(self.data['income_cat'] < 5, 5.0, inplace=True)
        for train_index, test_index in split.split(self.data, self.data['income_cat']):
            strat_train_set = self.data.iloc[train_index]
            strat_test_set = self.data.iloc[test_index]
            print("train set size {},test set size {}".format(len(strat_train_set), len(strat_test_set)))
        return strat_train_set, strat_test_set


if __name__ == '__main__':
    unittest.main()
