import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn_pandas.util import retain_sign, validate_dataframe


class DataFrameFixColumnOrder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def _validate_params(self):
        pass

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self._validate_params()
        self.columns = X.columns
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.loc[:, self.columns]


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
        X = X.copy()
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
        X = X.copy()
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

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self._validate_params()
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
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

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.astype(self.dtype)


class PrintToScreen(BaseEstimator, TransformerMixin):

    def __init__(self, max_rows=5, max_cols=10):
        self.max_rows = max_rows
        self.max_cols = max_cols

    def fit(self, X, y=None, **fitparams):
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        print(X.to_string(max_rows=self.max_rows, max_cols=self.max_cols))
        return X


class CreateDummyColumn(BaseEstimator, TransformerMixin):

    def __init__(self, value=0.0, column_name='__dummy__'):
        self.value = value
        self.column_name = column_name

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        X[self.column_name] = self.value
        return X


class InferType(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.types = {}

    def _validate_params(self, X=None):
        pass

    @staticmethod
    def _infer_dtype(x):
        return pd.api.types.infer_dtype(x, skipna=True)

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.types = {}
        for col in X.columns:
            self.types[col] = self._infer_dtype(X[col])
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.copy()


class ProfileDataFrame(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.col_attributes = {}

    def _validate_params(self, X=None):
        pass

    @staticmethod
    def _infer_dtype(x):
        return pd.api.types.infer_dtype(x, skipna=True)

    @staticmethod
    def _numeric_profile(x):
        return {
            'mean': np.nanmean(x),
            'std': np.nanstd(x),
            'median': np.nanmedian(x),
            'min': np.nanmin(x),
            'max': np.nanmax(x),
            'percentile_25': np.nanpercentile(x, 0.25),
            'percentile_75': np.nanpercentile(x, 0.75),
            'percent_missing': x.isna().mean(),
            'percent_infinite': np.isinf(x).mean(),
            'percent_finite': np.isfinite(x).mean(),
            'percent_unique': len(np.unique(x)) / len(x),
            'num_unique': len(np.unique(x))
        }

    @staticmethod
    def _string_profile(x):
        string_len_array = x.str.len()
        return {
            'most_frequent': x.value_counts().nlargest(n=1).index[0],
            'most_frequent_percent': x.value_counts().nlargest(n=1).values[0] / len(x),
            'avg_len': np.nanmean(string_len_array),
            'min_len': np.nanmin(string_len_array),
            'max_len': np.nanmax(string_len_array),
            'min': np.nanmin(x),
            'max': np.nanmax(x),
            'percent_missing': x.isna().mean(),
            'percent_blank': (x.str.strip() == '').mean(),
            'percent_unique': len(np.unique(x)) / len(x),
            'num_unique': len(np.unique(x))
        }

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.types = {}
        for col in X.columns:
            self.types[col] = self._infer_dtype(X[col])
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        return X.copy()