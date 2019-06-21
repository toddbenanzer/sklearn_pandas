import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn_pandas.util import retain_sign, validate_dataframe
from sklearn.decomposition import PCA, KernelPCA


class QuantileBinning(BaseEstimator, TransformerMixin):

    def __init__(self, nbins=5, prefix='', suffix=''):
        self.nbins = nbins
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.cuts = {}
        for col in X.columns:
            cuts = X[col].quantile(q=np.linspace(0 + 1 / self.nbins, 1 - 1 / self.nbins, self.nbins - 1)).tolist()
            self.cuts[col] = [-np.inf, ] + cuts[:] + [np.inf, ]
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
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
        X = validate_dataframe(X)
        self.clips = {}
        for col in X.columns:
            self.clips[col] = X[col].quantile(q=[self.clip_p, 1 - self.clip_p]).tolist()
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            X[new_col] = X[col].clip(self.clips[col][0], self.clips[col][1])

        return X.loc[:, new_col_list]


class PandasRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), prefix='', suffix=''):
        self.scaler = None
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.scaler = RobustScaler(with_centering=self.with_centering,
                               with_scaling=self.with_scaling, quantile_range=self.quantile_range)
        self.scaler.fit(X)
        self.center_ = pd.Series(self.scaler.center_, index=X.columns)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, with_mean=True, with_std=True, prefix='', suffix=''):
        self.scaler = None
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), copy=True, prefix='', suffix=''):
        self.scaler = None
        self.feature_range = feature_range
        self.copy = copy
        self.scale_ = None
        self.min_ = None
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.scaler = MinMaxScaler(feature_range=self.feature_range, copy=self.copy)
        self.scaler.fit(X)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        self.min_ = pd.Series(self.scaler.min_, index=X.columns)
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class MissingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='zero', create_indicators=False, indicator_only=False, prefix='', suffix=''):
        self.method = method
        self.create_indicators = create_indicators
        self.indicator_only = indicator_only
        self.prefix = prefix
        self.suffix = suffix

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
        X = validate_dataframe(X)
        self.impute_val = {}
        for col in X.columns:
            self.impute_val[col] = np.nan_to_num(self._calc_impute_val(X.loc[:, col]))
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xout = X.copy()
        Xout = Xout.replace([np.inf, -np.inf], np.nan)
        new_col_list = []
        for col in Xout.columns:
            new_col = self.prefix + col + self.suffix
            if not self.indicator_only:
                new_col_list.append(new_col)
            if self.create_indicators:
                Xout[col + '_isna'] = Xout[col].isna()
                new_col_list.append(col + '_isna')
            Xout[new_col] = Xout[col].fillna(self.impute_val[col])
        return Xout.loc[:, new_col_list]


class AggByGroupTransform(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_vars=[], metric_vars=[], agg_func='mean', delim='_'):
        self.groupby_vars = groupby_vars
        self.metric_vars = metric_vars
        self.agg_func = agg_func
        self.delim = delim

    def _validate_params(self, X):
        if self.agg_func == 'mean':
            self._agg_func = np.nanmean
        elif self.agg_func == 'min':
            self._agg_func = np.nanmin
        elif self.agg_func == 'max':
            self._agg_func = np.nanmax
        else:
            raise NotImplementedError("Did not implement {0} aggregation function".format(self.agg_func))

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self._validate_params(X)
        self.agg_series = {}
        for gb in self.groupby_vars:
            self.agg_series[gb] = {}
            for metric in self.metric_vars:
                agg_series = X.groupby(gb).agg({metric: self._agg_func})[metric]
                self.agg_series[gb][metric] = agg_series
        return self

    def _get_agg_val(self, gb, metric, x):
        try:
            return self.agg_series[gb][metric][x]
        except KeyError:
            return np.nan

    def transform(self, X):
        X = validate_dataframe(X)
        Xout = X.copy()
        new_col_list = []
        for gb in self.groupby_vars:
            for metric in self.metric_vars:
                new_col = metric + self.delim + self.agg_func + self.delim + 'by' + self.delim + gb
                new_col_list.append(new_col)
                #Xout[new_col] = self.agg_series[gb][metric].loc[X[gb]]
                Xout[new_col] = [self.agg_series[gb][metric][x] for x in X[gb]]

        return Xout.loc[:, new_col_list]


class PandasPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.9, copy=True, prefix='pca_', suffix=''):
        self.n_components = n_components
        self.copy = copy
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=True, with_std=True)
        self.scaler.fit(X)
        self.pca = PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(self.scaler.transform(X))
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xs = self.scaler.transform(X.copy())
        Xpca = self.pca.transform(Xs)
        column_names = [self.prefix + '{0:03g}'.format(n) + self.suffix for n in range(Xpca.shape[1])]
        return pd.DataFrame(Xpca, index=X.index, columns=column_names)


class PandasKernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kernel='linear', gamma=None, degree=3, copy=True, prefix='pca_', suffix=''):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.copy = copy
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=True, with_std=True)
        self.scaler.fit(X)
        self.kernelpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma, degree=self.degree)
        self.kernelpca.fit(self.scaler.transform(X))
        return self

    def transform(self, X):
        X = validate_dataframe(X)
        Xs = self.scaler.transform(X.copy())
        Xpca = self.kernelpca.transform(Xs)
        column_names = [self.prefix + '{0:03g}'.format(n) + self.suffix for n in range(Xpca.shape[1])]
        return pd.DataFrame(Xpca, index=X.index, columns=column_names)
