import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist, validate_dataframe


class StringImputer(BaseEstimator, TransformerMixin):

    def __init__(self, value_if_empty='Blank', value_if_none='None'):
        self.value_if_empty = value_if_empty
        self.value_if_none = value_if_none

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xout = X.copy()
        for col in Xout.columns:
            Xout[col][Xout[col].str.strip() == ''] = self.value_if_empty
            Xout[col][pd.isnull(Xout[col])] = self.value_if_none
        return Xout


class BundleRareValues(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.05, value_if_rare='Other'):
        self.threshold = threshold
        self.value_if_rare = value_if_rare

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.common_categories = {}
        for col in X.columns:
            counts = pd.Series(X[col].value_counts() / np.float(len(X)))
            self.common_categories[col] = list(counts[counts >= self.threshold].index)

        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xout = X.copy()
        for col in Xout.columns:
            Xout[col] = np.where(Xout[col].isin(
                self.common_categories[col]), Xout[col], self.value_if_rare)

        return Xout


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delim='_'):
        self.delim = delim

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.encodings = {}
        for col in X.columns:
            self.encodings[col] = np.sort(X[col].unique())

        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xout = X.copy()
        new_col_list = []
        for col in X.columns:
            for cat in self.encodings[col]:
                new_col = col + '_' + str(cat)
                Xout[new_col] = Xout[col] == cat
                new_col_list.append(new_col)

        return Xout.loc[:, new_col_list]
