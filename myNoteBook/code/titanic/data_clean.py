from sklearn.base import BaseEstimator, TransformerMixin


class NameToLength(BaseEstimator, TransformerMixin):

    @property
    def add_bedrooms_per_room(self):
        return self.__add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['NameLen'] = X["Name"].apply(len)
        return X.drop("Name", axis=1)
