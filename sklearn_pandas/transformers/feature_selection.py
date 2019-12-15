import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn_pandas.util import retain_sign, validate_dataframe
from sklearn.base import BaseEstimator, TransformerMixin, clone


class PandasSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        X = validate_dataframe(X)
        imp = {}
        for col in X.columns:

        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        return X