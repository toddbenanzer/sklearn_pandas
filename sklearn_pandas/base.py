import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn_pandas.util import retain_sign, validate_dataframe


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        df_concat = pd.concat([t.transform(X) for t in self.fitted_transformers_], axis=1).copy()
        return df_concat


class DataFrameModelTransformer(TransformerMixin):

    def __init__(self, model, output_column_names=None):
        self.model = model
        self.output_column_names = output_column_names

    def _validate_params(self):
        if self.output_column_names is None:
            self.output_column_names = ['model_out']
        elif type(self.output_column_names) == str:
            self.output_column_names = [self.output_column_names, ]

    def fit(self, *args, **kwargs):
        self._validate_params()
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        X = validate_dataframe(X)
        y = self.model.predict(X)
        return pd.DataFrame(y, columns=self.output_column_names)


class DataFrameFunctionApply(BaseEstimator, TransformerMixin):

    def __init__(self, func=None, prefix='', suffix='', safe_sign=False):
        self.func = func
        self.prefix = prefix
        self.suffix = suffix
        self.safe_sign = safe_sign

    def _validate_params(self):
        if self.func is None:
            self.func = lambda x: x

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params()
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        new_col_list = []
        for col in X.columns:
            new_col_name = self.prefix + col + self.suffix
            new_col_list.append(new_col_name)
            if self.safe_sign:
                X[new_col_name] = X[col].map(retain_sign(self.func))
            else:
                X[new_col_name] = X[col].map(self.func)
        return X.loc[:, new_col_list]


class TypeCast(BaseEstimator, TransformerMixin):

    def __init__(self, dtype=float):
        self.dtype = dtype

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        return X.astype(self.dtype)


class PrintToScreen(BaseEstimator, TransformerMixin):

    def __init__(self, max_rows=5, max_cols=10):
        self.max_rows = max_rows
        self.max_cols = max_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        print(X.to_string(max_rows=self.max_rows, max_cols=self.max_cols))
        return X


class CreateDummyColumn(BaseEstimator, TransformerMixin):

    def __init__(self, value=0.0, column_name='__dummy__'):
        self.value = value
        self.column_name = column_name

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xt = X.copy()
        Xt[self.column_name] = self.value
        return Xt
