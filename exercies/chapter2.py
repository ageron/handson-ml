import os
import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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


def display_score(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names) -> None:
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


class CombinedAttributesAddr(BaseEstimator, TransformerMixin):
    room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True) -> None:
        self.__add_bedrooms_per_room = add_bedrooms_per_room

    @property
    def add_bedrooms_per_room(self):
        return self.__add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.room_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.__add_bedrooms_per_room:
            bedroom_per_room = X[:, self.bedroom_ix] / X[:, self.room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class TestHouseAnalysis(TestCase):

    def setUp(self) -> None:
        self.data = load_housing_data()
        self.__pipeline = None
        self.labels_column = "median_house_value"

    @property
    def pipeline(self):
        if self.__pipeline is None:
            self.__pipeline = self.create_pipeline()
            train_set, _ = self.split_analysis_set()
            drop = train_set.drop(self.labels_column, axis=1)
            self.pipeline.fit(drop)
        return self.__pipeline

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

    def test_data_relationship(self):
        data = load_housing_data()
        corr_matrix = data.corr()
        print(corr_matrix['median_house_value'].sort_values(ascending=False))
        attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
        scatter_matrix(data[attributes], figsize=(12, 8))
        data.plot(kind="scatter", x='median_income', y='median_house_value', alpha=0.1)
        plt.show()

        print("new corr:")
        print(self.create_new_column().corr()['median_house_value'].sort_values(ascending=False))

        addr = CombinedAttributesAddr()
        print(addr.transform(data.values))
        self.assertTrue(True)

    def test_prepare_ml_data(self):
        train_set, _ = self.split_analysis_set()
        imputer = SimpleImputer(strategy='median')
        housing = train_set.drop("median_house_value", axis=1)
        housing_lab = train_set['median_house_value'].copy()
        housing_num = housing.drop('ocean_proximity', axis=1)
        imputer.fit(housing_num)
        print(housing_num.head())
        print(imputer.statistics_)
        housing_tr = pd.DataFrame(imputer.transform(housing_num), columns=housing_num.columns)
        print(housing_tr.head())

    def test_handle_text_data(self):
        encoder = LabelEncoder()
        data = load_housing_data()
        housing_cat = data['ocean_proximity']
        housing_cat_encoded = encoder.fit_transform(housing_cat)
        print("source data head:")
        print(housing_cat.head())
        print("encoder data head:")
        print(housing_cat_encoded)
        print(encoder.classes_)

        one_hot_encoder = OneHotEncoder()
        one_hot_cat = one_hot_encoder.fit_transform(data[['ocean_proximity']])
        print(one_hot_cat.toarray())

    def test_add_pipeline(self):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAddr()),
            ('std_scaler', StandardScaler())
        ])
        train_set, _ = self.split_analysis_set()
        housing = train_set.drop("median_house_value", axis=1)
        housing_num = housing.drop('ocean_proximity', axis=1)

        housing_num_tr = num_pipeline.fit_transform(housing_num)
        print(housing_num_tr)

    def test_prepare_data_for_ML(self):
        housing_prepared, _, _ = self.prepare_data_set()

        print(housing_prepared)

    def test_analysis_data(self):
        housing = self.data.copy()
        train_set, test_set, labels = self.prepare_data_set()
        lin_reg = LinearRegression()
        lin_reg.fit(train_set, labels)

        housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
        housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
        some_data = test_set.iloc[:5]
        some_labels = labels.iloc[:5]
        pipeline = self.pipeline
        data = pipeline.transform(some_data)
        print("predict:")
        predict = lin_reg.predict(data)
        print(predict)

        print("mean squared error!!!")
        print(np.sqrt(mean_squared_error(some_labels, predict)))

        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(train_set, labels)

        tree_predict = tree_reg.predict(data)
        print("tree predict:")
        print(tree_predict)

        print("tree sqrt error!!!")
        print(np.sqrt(mean_squared_error(some_labels, tree_predict)))

        print("decision tree regress")
        display_score(np.sqrt(
            -cross_val_score(DecisionTreeRegressor(), train_set, labels, scoring='neg_mean_squared_error', cv=10)))
        print("line regress")
        display_score(np.sqrt(
            -cross_val_score(LinearRegression(), train_set, labels, scoring='neg_mean_squared_error', cv=10)))
        print("random Forest regress")
        display_score(np.sqrt(
            -cross_val_score(RandomForestRegressor(), train_set, labels, scoring='neg_mean_squared_error', cv=10)))

    def test_grid_search(self):
        param_grid = [
            {
                'n_estimators': [3, 10, 30],
                'max_features': [2, 4, 6, 9]
            },
            {
                'bootstrap': [False],
                'n_estimators': [3, 10],
                'max_features': [2, 3, 4]
            }
        ]

        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        train_set, _, labels = self.prepare_data_set()
        grid_search.fit(train_set, labels)
        print(grid_search.best_estimator_)
        print(grid_search.best_params_)
        # print(grid_search.cv_results_)
        feature_importances = grid_search.best_estimator_.feature_importances_
        print(len(feature_importances))

        housing_num = self.data.drop('ocean_proximity', axis=1).drop(self.labels_column, axis=1)
        num_attribs = list(housing_num)
        extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
        # cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
        cat_encoder = self.pipeline.named_transformers_["cat"]
        cat_one_hot_attribs = list(cat_encoder.categories_[0])
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        print((len(attributes)))
        print(sorted(zip(feature_importances, attributes), reverse=True))

    def prepare_data_set(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        train_set, test_set = self.split_analysis_set()
        housing_labels = train_set[self.labels_column].copy()
        train_set = train_set.drop(self.labels_column, axis=1)
        test_set = test_set.drop(self.labels_column, axis=1)
        full_pipeline = self.pipeline
        housing_prepared = full_pipeline.transform(train_set)
        return housing_prepared, test_set, housing_labels

    def create_pipeline(self) -> Pipeline:
        housing_num = self.data.drop('ocean_proximity', axis=1).drop(self.labels_column, axis=1)
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAddr()),
            ('std_scaler', StandardScaler())
        ])
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        return full_pipeline

    def create_new_column(self) -> pd.DataFrame:
        data = self.data.copy()
        data['rooms_per_household'] = data['total_rooms'] / data['households']
        data['bed_per_rooms'] = data['total_bedrooms'] / data['total_rooms']
        data['population_per_household'] = data['population'] / data['households']
        return data

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
