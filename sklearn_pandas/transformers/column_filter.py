import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist, validate_dataframe
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def _validate_params(self, X):
        if self.columns is None:
            self.columns = X.columns
        elif type(self.columns) is str:
            self.columns = [self.columns, ]

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params(X)
        validate_columns_exist(X, self.columns)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
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
        X = validate_dataframe(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
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
        X = validate_dataframe(X)
        self._validate_params(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        selected_columns = [col for col in X.columns if col not in self.drop_columns]
        return X.loc[:, selected_columns]


class UniqueValueFilter(BaseEstimator, TransformerMixin):

    def __init__(self, min_unique_values=2):
        self.min_unique_values = min_unique_values

    def _validate_params(self, X):
        pass

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.drop_columns = [col for col in X.columns if X[col].nunique() < self.min_unique_values]
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        selected_columns = [col for col in X.columns if col not in self.drop_columns]
        return X.loc[:, selected_columns]


class ColumnByType(BaseEstimator, TransformerMixin):

    def __init__(self, numerics=False, strings=False, dates=False, booleans=False, var_types=None):
        self.numerics = numerics
        self.strings = strings
        self.dates = dates
        self.booleans = booleans
        self.var_types = var_types

    def _validate_params(self, X):
        self.selected_var_types = []
        if self.numerics:
            self.selected_var_types.extend(['integer', 'mixed-integer-float', 'floating', 'decimal', ])
        elif self.strings:
            self.selected_var_types.extend(['string', 'bytes', 'mixed-integer', 'categorical', ])
        elif self.booleans:
            self.selected_var_types.extend(['boolean', ])
        elif self.dates:
            self.selected_var_types.extend(['datetime', 'date', 'datetime64', 'timedelta', ])
        else:
            self.selected_var_types.extend(self.var_types)

    @staticmethod
    def _infer_dtype(x):
        return pd.api.types.infer_dtype(x, skipna=True)

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.selected_columns = X.columns[X.apply(self._infer_dtype).isin(self.selected_var_types)]
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        return X.loc[:, self.selected_columns]


class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.95, method='pearson'):
        self.threshold = threshold
        self.method = method

    def _validate_params(self, X):
        pass

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params(X)
        corr_matrix = X.corr(self.method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        self.drop_columns = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        selected_columns = [col for col in X.columns if col not in self.drop_columns]
        return X.loc[:, selected_columns]


class PandasSelectKBest(BaseEstimator, TransformerMixin):

    def __init__(self, score_func=f_regression, k=10, ascending=True):
        self.score_func = score_func
        self.k = k
        self.ascending = ascending

    def _validate_params(self, X):
        pass

    def fit(self, X, y):
        X = validate_dataframe(X)
        self._validate_params(X)
        var_performance = self.score_func(X, y)
        if type(var_performance) == tuple:
            var_performance = pd.Series(data=var_performance[-1], index=X.columns)
        self.k_best_columns = var_performance.sort_values(ascending=self.ascending).head(self.k).index
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        X = X.copy()
        selected_columns = [col for col in X.columns if col in self.k_best_columns]
        return X.loc[:, selected_columns]
