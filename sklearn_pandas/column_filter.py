import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def _validate_params(self, X):
        if self.columns is None:
            self.columns = X.columns
        elif type(self.columns) is str:
            self.columns = [self.columns, ]

    def fit(self, X, y=None):
        self._validate_params(X)
        validate_columns_exist(X, self.columns)
        return self

    def transform(self, X):
        validate_columns_exist(X, self.columns)
        return X.loc[:, self.columns]


class ColumnSearchSelect(BaseEstimator, TransformerMixin):

    def __init__(self, contains=None, prefix=None, suffix=None, operator='and'):
        self.contains = contains
        self.prefix = prefix
        self.suffix = suffix
        self.operator = operator

    def _validate_params(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_columns = len(X.columns)

        if self.contains is None:
            contains_mask = [True] * num_columns
        else:
            contains_mask = X.columns.str.contains(self.contains)

        if self.prefix is None:
            prefix_mask = [True] * num_columns
        else:
            prefix_mask = X.columns.str.startswith(self.prefix)

        if self.suffix is None:
            suffix_mask = [True] * num_columns
        else:
            suffix_mask = X.columns.str.endswith(self.suffix)

        if self.operator == 'and':
            selected_columns = [c and p and s for c, p, s in zip(contains_mask, prefix_mask, suffix_mask)]
        elif self.operator == 'or':
            selected_columns = [c or p or s for c, p, s in zip(contains_mask, prefix_mask, suffix_mask)]
        else:
            raise ValueError("Operator {0} is not implemented".format(self.operator))

        return X.loc[:, selected_columns]


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.drop_columns = columns

    def _validate_params(self, X):
        if self.drop_columns is None:
            self.drop_columns = []
        elif type(self.drop_columns) is str:
            self.drop_columns = [self.drop_columns, ]

    def fit(self, X, y=None):
        self._validate_params(X)
        return self

    def transform(self, X):
        selected_columns = [col for col in X.columns if col not in self.drop_columns]
        return X.loc[:, selected_columns]
