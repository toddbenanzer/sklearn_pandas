import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway, spearmanr, kendalltau
from sklearn_pandas.util import retain_sign, validate_dataframe
from sklearn.base import BaseEstimator, TransformerMixin, clone


class PandasSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, method='pearson'):
        self.k = k
        self.method = method

    def _check_params(self, X, y):
        num_cols = len(X.columns)
        if type(self.k) == float:
            self._k = int(np.ceil(num_cols * self.k))
        elif (type(self.k) == str) & (self.k == 'all'):
            self._k = num_cols
        if type(self.k) == int:
            self._k = self.k
        self._k = min(self._k, num_cols)
        self._k = max(self._k, 1)

    def calc_rank(self, X, y):
        if self.method == 'ftest':
            corr = 1.0 * X.apply(func=lambda x, y: f_oneway(x, y)[0], axis=0, args=(y,)).abs()
            return corr.rank(method='min')

        elif self.method == 'pearson':
            corr = -1.0 * X.apply(func=lambda x, y: pearsonr(x, y)[0], axis=0, args=(y,)).abs()
            return corr.rank(method='min')

        elif self.method == 'spearman':
            corr = -1.0 * X.apply(func=lambda x, y:spearmanr(x, y)[0], axis=0, args=(y,)).abs()
            return corr.rank(method='min')

        elif self.method == 'spearman':
            corr = -1.0 * X.apply(func=lambda x, y: kendalltau(x, y)[0], axis=0, args=(y,)).abs()
            return corr.rank(method='min')

    def fit(self, X, y, **fitparams):
        X = validate_dataframe(X)
        self._check_params(X, y)
        self.rank = self.calc_rank(X, y)
        self.selected_columns = self.rank[self.rank <= self._k].index.tolist()
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.loc[:, self.selected_columns]


class PandasSelectThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, pct=0.05, method='pearson'):
        self.pct = pct
        self.method = method

    def _check_params(self, X, y):
        num_cols = len(X.columns)
        self._pct = self.pct
        self._pct = min(self._pct, 1.0)
        self._pct = max(self._pct, 0.0)

    def calc_imp(self, X, y):
        if self.method == 'ftest':
            corr = X.apply(func=lambda x, y: f_oneway(x, y)[0], axis=0, args=(y,)).abs()
            return corr

        elif self.method == 'pearson':
            corr = X.apply(func=lambda x, y: pearsonr(x, y)[0], axis=0, args=(y,)).abs()
            return corr

        elif self.method == 'spearman':
            corr = X.apply(func=lambda x, y:spearmanr(x, y)[0], axis=0, args=(y,)).abs()
            return corr

        elif self.method == 'kendalltau':
            corr = X.apply(func=lambda x, y: kendalltau(x, y)[0], axis=0, args=(y,)).abs()
            return corr

    def fit(self, X, y, **fitparams):
        X = validate_dataframe(X)
        self._check_params(X, y)
        self.imp = self.calc_imp(X, y)
        self.selected_columns = self.imp[self.imp >= self._pct].index.tolist()
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.loc[:, self.selected_columns]
