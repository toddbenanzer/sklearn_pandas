import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist, validate_dataframe


class DropNARowFilter(BaseEstimator, TransformerMixin):

    def __init__(self, excluded_columns=None):
        self.excluded_columns = excluded_columns or []

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        return self

    def transform(self, X, y=None):
        X = validate_dataframe(X)
        X = X.copy()
        subset = [c for c in X.columns if c not in self.excluded_columns]
        return X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any', inplace=False, subset=subset)
