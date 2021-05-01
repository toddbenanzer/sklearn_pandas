import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist, validate_dataframe


class StringImputer(BaseEstimator, TransformerMixin):

    def __init__(self, value_if_empty='Blank', value_if_none='None'):
        self.value_if_empty = value_if_empty
        self.value_if_none = value_if_none

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        return self

    def transform(self, X, **transformparams):
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

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        if 'sample_weight' in fitparams:
            w = pd.Series(fitparams['sample_weight'])
        else:
            w = pd.Series(np.ones(len(y)))
        
        self.common_categories = {}
        for col in X.columns:
            #counts = pd.Series(X[col].value_counts() / float(len(X)))
            counts = w.groupby(X[col]).agg('sum') / w.sum()
            self.common_categories[col] = list(counts[counts >= self.threshold].index)

        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        Xout = X.copy()
        for col in Xout.columns:
            Xout[col] = np.where(Xout[col].isin(
                self.common_categories[col]), Xout[col], self.value_if_rare)

        return Xout


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delim='_'):
        self.delim = delim

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.encodings = {}
        for col in X.columns:
            self.encodings[col] = np.sort(X[col].unique())

        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        Xout = X.copy()
        new_col_list = []
        for col in X.columns:
            for cat in self.encodings[col]:
                new_col = col + '_' + str(cat)
                Xout[new_col] = Xout[col] == cat
                new_col_list.append(new_col)

        return Xout.loc[:, new_col_list]


class CategoricalAggregate(BaseEstimator, TransformerMixin):
    def __init__(self, agg_func='mean', rank=False, prefix='', suffix='__avg_resp'):
        self.agg_func = agg_func
        self.rank = rank
        self.prefix = prefix
        self.suffix = suffix


    def fit(self, X, y, **fitparams):
        X = validate_dataframe(X)

        if 'sample_weight' in fitparams:
            w = fitparams['sample_weight']
        else:
            w = pd.Series(np.ones(len(y)))

        def weighted_mean(x): 
            return np.average(x, weights=w.loc[x.index])

        def _weighted_median(x, weights):
            x_ = x.copy()
            weights_ = weights.copy()
            _i_sorted = x_.index[np.argsort(x_)]
            x_sorted = x_[_i_sorted]
            w_sorted = weights_[_i_sorted]
            cumsum = w_sorted.cumsum() / w_sorted.sum()
            print('cumsum', cumsum)
            print('x_sorted', cumsum)
            return np.interp(0.5, cumsum, x_sorted)

        def weighted_median(x): 
            return _weighted_median(x, weights=w.loc[x.index])

        if self.agg_func == 'mean':
            self._agg_func = weighted_mean
        elif self.agg_func == 'median':
            self._agg_func = weighted_median
        elif self.agg_func == 'min':
            self._agg_func = np.min
        elif self.agg_func == 'max':
            self._agg_func = np.max
        else:
            raise NotImplementedError('Aggreagation Function {0} is not implemented'.format(self.agg_func))

        self.agg_series = {}
        for col in X.columns:
            if self.rank:
                self.agg_series[col] = y.groupby(X[col]).agg({self._agg_func}).rank().iloc[:, 0]
            else:
                self.agg_series[col] = y.groupby(X[col]).agg({self._agg_func}).iloc[:,0]
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        Xout = X.copy()
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            Xout[new_col] = [self.agg_series[col][x] for x in X[col]]
        return Xout.loc[:, new_col_list]


class IntegerToString(BaseEstimator, TransformerMixin):

    def __init__(self, min_unique_values=5):
        self.min_unique_values = min_unique_values
        self.hidden_categorical_columns = []

    def _validate_params(self, X):
        pass

    @staticmethod
    def _infer_dtype(x):
        return pd.api.types.infer_dtype(x, skipna=True)

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.hidden_categorical_columns = []
        for col in X.columns:
            is_integer = self._infer_dtype(X[col]) in ['integer', 'mixed-integer', ]
            if is_integer:
                num_unique = X[col].nunique()
                if num_unique <= self.min_unique_values:
                    self.hidden_categorical_columns.append(col)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        for col in self.hidden_categorical_columns:
            X[col] = X[col].astype(str)
        return X

