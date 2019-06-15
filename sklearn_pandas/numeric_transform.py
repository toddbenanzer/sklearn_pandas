import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn_pandas.util import retain_sign


class QuantileBinning(BaseEstimator, TransformerMixin):

    def __init__(self, nbins=5, prefix='', suffix=''):
        self.nbins = nbins
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        self.cuts = {}
        for col in X.columns:
            cuts = X[col].quantile(q=np.linspace(0 + 1 / self.nbins, 1 - 1 / self.nbins, self.nbins - 1)).tolist()
            self.cuts[col] = [-np.inf, ] + cuts[:] + [np.inf, ]
        return self

    def transform(self, X, **transformparams):
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            X[new_col] = pd.cut(x=X[col], bins=self.cuts[col], duplicates='drop')

        return X.loc[:, new_col_list]


class WinsorizeTransform(BaseEstimator, TransformerMixin):

    def __init__(self, clip_p, prefix='', suffix=''):
        self.clip_p = clip_p
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        self.clips = {}
        for col in X.columns:
            self.clips[col] = X[col].quantile(q=[self.clip_p, 1 - self.clip_p]).tolist()
        return self

    def transform(self, X, **transformparams):
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            X[new_col] = X[col].clip(self.clips[col][0], self.clips[col][1])

        return X.loc[:, new_col_list]


class PandasRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        self.scaler = None
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.scaler = RobustScaler(with_centering=self.with_centering,
                               with_scaling=self.with_scaling, quantile_range=self.quantile_range)
        self.scaler.fit(X)
        self.center_ = pd.Series(self.scaler.center_, index=X.columns)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        return self

    def transform(self, X):
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.scaler = None
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.scale_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        self.scaler.fit(X)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        self.mean_ = pd.Series(self.scaler.mean_, index=X.columns)
        self.var_ = pd.Series(self.scaler.var_, index=X.columns)
        return self

    def transform(self, X):
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), copy=True):
        self.scaler = None
        self.feature_range = feature_range
        self.copy = copy
        self.scale_ = None
        self.min_ = None

    def fit(self, X, y=None):
        self.scaler = MinMaxScaler(feature_range=self.feature_range, copy=self.copy)
        self.scaler.fit(X)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        self.min_ = pd.Series(self.scaler.min_, index=X.columns)
        return self

    def transform(self, X):
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled


class MissingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='zero', create_indicators=False):
        self.method = method
        self.create_indicators = create_indicators

    def _calc_impute_val(self, x):
        if self.method == 'zero':
            return 0
        elif self.method == 'mean':
            return np.nanmean(x.replace([np.inf, -np.inf], np.nan))
        elif self.method == 'median':
            return np.nanmedian(x.replace([np.inf, -np.inf], np.nan))
        else:
            raise NotImplementedError('method {0} not implemented'.format(self.method))

    def fit(self, X, y=None):
        self.impute_val = {}
        for col in X.columns:
            self.impute_val[col] = np.nan_to_num(self._calc_impute_val(X.loc[:, col]))
        return self

    def transform(self, X):
        Xout = X.copy()
        Xout = Xout.replace([np.inf, -np.inf], np.nan)
        for col in Xout.columns:
            if self.create_indicators:
                Xout[col + '_isna'] = Xout[col].isna()
            Xout[col] = Xout[col].fillna(self.impute_val[col])
        return Xout
