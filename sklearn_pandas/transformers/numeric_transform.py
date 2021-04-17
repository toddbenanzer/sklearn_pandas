import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn_pandas.util import retain_sign, validate_dataframe
from sklearn.decomposition import PCA, KernelPCA


class QuantileBinning(BaseEstimator, TransformerMixin):

    def __init__(self, nbins=5, prefix='', suffix='__qbin'):
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
        X = X.copy()
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            X[new_col] = pd.cut(x=X[col], bins=self.cuts[col], duplicates='drop')

        return X.loc[:, new_col_list]


class WinsorizeTransform(BaseEstimator, TransformerMixin):

    def __init__(self, clip_p, prefix='', suffix='__wins'):
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
        X = X.copy()
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            new_col_list.append(new_col)
            X[new_col] = X[col].clip(self.clips[col][0], self.clips[col][1])

        return X.loc[:, new_col_list]


class PandasRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), prefix='', suffix='__rbstscale'):
        self.scaler = None
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.scaler = RobustScaler(with_centering=self.with_centering,
                               with_scaling=self.with_scaling, quantile_range=self.quantile_range)
        self.scaler.fit(X)
        self.center_ = pd.Series(self.scaler.center_, index=X.columns)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, with_mean=True, with_std=True, prefix='', suffix='__stdscale'):
        self.scaler = None
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        self.scaler.fit(X)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), copy=True, prefix='', suffix='__mmscale'):
        self.scaler = None
        self.feature_range = feature_range
        self.copy = copy
        self.scale_ = None
        self.min_ = None
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.scaler = MinMaxScaler(feature_range=self.feature_range, copy=self.copy)
        self.scaler.fit(X)
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        self.min_ = pd.Series(self.scaler.min_, index=X.columns)
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        Xrs = self.scaler.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=self.prefix + X.columns + self.suffix)
        return Xscaled


class MissingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='zero', create_indicators=False, indicator_only=False, prefix='', suffix='__impute'):
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

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.impute_val = {}
        for col in X.columns:
            self.impute_val[col] = np.nan_to_num(self._calc_impute_val(X.loc[:, col]))
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            if not self.indicator_only:
                new_col_list.append(new_col)
            if self.create_indicators or self.indicator_only:
                X[col + '_isna'] = X[col].isna()
                new_col_list.append(col + '_isna')
            X[new_col] = X[col].fillna(self.impute_val[col])
        return X.loc[:, new_col_list]


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
        elif self.agg_func == 'median':
            self._agg_func = np.nanmedian
        else:
            raise NotImplementedError("Did not implement {0} aggregation function".format(self.agg_func))

    def fit(self, X, y=None, **fitparams):
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

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        new_col_list = []
        for gb in self.groupby_vars:
            for metric in self.metric_vars:
                new_col = metric + self.delim + self.agg_func + self.delim + 'by' + self.delim + gb
                new_col_list.append(new_col)
                #X[new_col] = self.agg_series[gb][metric].loc[X[gb]]
                X[new_col] = [self.agg_series[gb][metric][x] for x in X[gb]]

        return X.loc[:, new_col_list]


class PandasPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.9, copy=True, prefix='pca_', suffix=''):
        self.n_components = n_components
        self.copy = copy
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=True, with_std=True)
        self.scaler.fit(X)
        self.pca = PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(self.scaler.transform(X))
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        Xs = self.scaler.transform(X)
        Xpca = self.pca.transform(Xs)
        column_names = [self.prefix + '{0:03g}'.format(n) + self.suffix for n in range(Xpca.shape[1])]
        return pd.DataFrame(Xpca, index=X.index, columns=column_names)


class PandasKernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kernel='linear', gamma=None, degree=3, copy=True, prefix='kpca_', suffix=''):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.copy = copy
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.scaler = StandardScaler(copy=self.copy, with_mean=True, with_std=True)
        self.scaler.fit(X)
        self.kernelpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma, degree=self.degree)
        self.kernelpca.fit(self.scaler.transform(X))
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        Xs = self.scaler.transform(X)
        Xpca = self.kernelpca.transform(Xs)
        column_names = [self.prefix + '{0:03g}'.format(n) + self.suffix for n in range(Xpca.shape[1])]
        return pd.DataFrame(Xpca, index=X.index, columns=column_names)


class PandasOutlierTrim(BaseEstimator, TransformerMixin):
    def __init__(self, method='IQR', range=1.5, values=True, indicators=True, prefix='', suffix='__outtrm',
                 low_pct=0.25, up_pct=0.75):
        self.method = method
        self.range = range
        self.values = values
        self.indicators = indicators
        self.low_pct = low_pct
        self.up_pct = up_pct
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        q1 = X.quantile(self.low_pct)
        q3 = X.quantile(self.up_pct)
        iqr = q3 - q1
        self.lb = q1 - self.range * iqr
        self.ub = q3 + self.range * iqr
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            x_orig = X[col].copy()
            if self.values:
                X[new_col] = x_orig
                X.loc[(X[new_col] < self.lb[col]), new_col] = self.lb[col]
                X.loc[(X[new_col] > self.ub[col]), new_col] = self.ub[col]
                new_col_list.append(new_col)
            if self.indicators:
                X[col + '_isoutlier'] = 0.0
                X.loc[x_orig < self.lb[col], col + '_isoutlier'] = 1.0
                X.loc[x_orig > self.ub[col], col + '_isoutlier'] = 1.0
                new_col_list.append(col + '_isoutlier')
        return X.loc[:, new_col_list]


class EntropyBinning(BaseEstimator, TransformerMixin):
    def __init__(
        self, method='entropy', min_gain=0.01, max_bins=10, noise_penalty=0.0, prefix='', suffix='__binned',
        nhist=50, min_pop=0.10, max_cuts=10):
        self.method = method
        self.min_gain = min_gain
        self.max_bins = max_bins
        self.noise_penalty = noise_penalty
        self.prefix = prefix
        self.suffix = suffix
        self.nhist = nhist
        self.min_pop = min_pop
        self.max_cuts = max_cuts

    def _round_cuts(self, cuts):

        def string_round(x, n=0):
            template_str = '{0:.' + str(n) + 'f}'
            return template_str.format(x)

        n = len(cuts)
        for r in range(20):
            rounded_cuts = np.unique(np.round(cuts, r))
            if len(rounded_cuts) == n:
                return [string_round(x, n=r) for x in cuts]

        return [string_round(x, n=20) for x in cuts]

    def _create_bin_labels(self, cuts):
        rounded_cuts = self._round_cuts(cuts)
        return ['{0}-{1}'.format(left, right) for left, right in zip(rounded_cuts[:-1], rounded_cuts[1:])]

    def _apply_bins(self, x, cuts):
        x_out = pd.cut(x, cuts, right=True, labels=self._create_bin_labels(cuts), retbins=False, include_lowest=False, duplicates='raise')
        x_out = x_out.astype(str)
        x_out = np.where(x_out == 'nan', 'Unknown', x_out)
        return x_out

    def _create_contingency_table(self, x, y, w):
        _df = pd.DataFrame({
            'x': x.values, 'y': y.values, 'w': w.values,
            }, index=[list(range(len(x)))])
        _df['q'] = pd.qcut(_df['x'], self.nhist, duplicates='drop').astype(str)

        def gb_wt_avg(x):
            return np.average(x, weights = _df.loc[x.index, 'w'])

        base_df = _df \
            .groupby('q', as_index=False) \
            .agg({'x': gb_wt_avg, 'y': gb_wt_avg, 'w': np.nansum})

        nan_mask = np.isnan(x)
        if sum(nan_mask) == 0:
            nan_df = pd.DataFrame({
                'q': ['Missing'], 
                'x': [np.nan], 
                'y': [0.0], 
                'w': [0.0]
            })
        else:
            nan_df = pd.DataFrame({
                'q': ['Missing'], 
                'x': [np.nan], 
                'y': [np.average(y[nan_mask], weights=w[nan_mask])], 
                'w': [np.nansum(w[nan_mask])]
            })
        
        contingency_df = pd.concat([
                nan_df,
                base_df
            ]).set_index('q')

        contingency_df['w'] = contingency_df['w'].clip(lower=0.0001, upper=None)
        
        return contingency_df

    def _eval_cuts(self, contingency, cuts):
        binned_df = contingency.copy()
        binned_df['xb'] = self._apply_bins(binned_df['x'], cuts=cuts)

        def wtd_cov(x):
            average = np.average(x, weights = binned_df.loc[x.index, 'w'])
            variance = np.average((x-average)**2, weights = binned_df.loc[x.index, 'w'])
            return np.divide(np.sqrt(variance), average)
            
        def wtd_var(x):
            average = np.average(x, weights = binned_df.loc[x.index, 'w'])
            variance = np.average((x-average)**2, weights = binned_df.loc[x.index, 'w'])
            return np.sqrt(variance)
        
        def _entropy(p):
            p = min(0.99999, p)
            p = max(0.00001, p)
            return -p * np.log(p)

        def wtd_entropy(x):
            average = np.average(x, weights = binned_df.loc[x.index, 'w'])
            return _entropy(average)

        if self.method == 'entropy':
            eval_func = wtd_entropy
        elif self.method == 'variance':
            eval_func = wtd_var
        else:
            raise NotImplementedError('Error {0} is not recognized'.format(self.method))

        bgb_df = binned_df.groupby('xb').agg({'y': eval_func, 'w': np.sum}).fillna(0)

        avg_std = np.average(bgb_df['y'], weights=bgb_df['w'])

        wts = bgb_df.loc[bgb_df.index != 'Unknown', 'w']
        min_population = np.min(wts / np.sum(wts))

        return avg_std, min_population

    def _calc_optimal_cuts(self, contingency):
        all_cuts = contingency['x'].dropna().tolist()[1:-1]

        # initialize optimization
        best_cuts = [-np.inf, np.inf]
        best_eval, best_min_pop = self._eval_cuts(contingency, cuts=best_cuts)
        curr_best_cuts = best_cuts[:]
        curr_best_eval = best_eval
        # find optimal cuts
        for iter in range(self.max_cuts):
            next_best_cuts = curr_best_cuts[:]
            next_best_eval = curr_best_eval
            for cut in all_cuts:
                new_cuts = list(np.sort(np.unique(curr_best_cuts + [cut])))
                new_eval, new_min_pop = self._eval_cuts(contingency, cuts=new_cuts)
                if new_eval < next_best_eval and new_min_pop >= self.min_pop:
                    next_best_cuts = new_cuts[:]
                    next_best_eval = new_eval
                    eval_increase = next_best_eval - curr_best_eval
            # update current
            curr_best_cuts = next_best_cuts[:]
            curr_best_eval = next_best_eval
        return list(curr_best_cuts)

    def fit(self, X, y, **fitparams):
        X = validate_dataframe(X)
        self.cuts = {}
        if 'sample_weight' in fitparams:
            w = fitparams['sample_weight']
        else:
            w = pd.Series(np.ones(len(y)))
        for col in X.columns:
            _cont_df = self._create_contingency_table(X[col], y, w)
            cuts = self._calc_optimal_cuts(_cont_df)
            self.cuts[col] = cuts[:]
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        new_col_list = []
        for col in X.columns:
            new_col = self.prefix + col + self.suffix
            x_orig = X[col].copy()
            X[new_col] = self._apply_bins(x_orig, self.cuts[col])
            new_col_list.append(new_col)

        return X.loc[:, new_col_list]