import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def fit(self, X, y=None, **fitparams):
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self

    def transform(self, X, **transformparams):
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
        y = self.model.predict(X)
        return pd.DataFrame(y, columns=self.output_column_names)


class DataFrameFunctionApply(BaseEstimator, TransformerMixin):

    def __init__(self, func=None, prefix='', suffix=''):
        self.func = func
        self.prefix = prefix
        self.suffix = suffix

    def _validate_params(self):
        if self.func is None:
            self.func = lambda x: x

    def fit(self, X, y=None):
        self._validate_params()
        return self

    def transform(self, X):
        new_col_list = []
        for col in X.columns:
            new_col_name = self.prefix + col + self.suffix
            new_col_list.append(new_col_name)
            X[new_col_name] = X[col].map(self.func)
        return X.loc[:, new_col_list]
