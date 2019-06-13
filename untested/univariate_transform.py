

class NumBinner(BaseEstimator, TransformerMixin):
    """Binnig data in specified numerical columns values using 3 uniform quantiles (from 0 to 1).
    Usage example:
        `new_cat_cols, new_num_cols, modified_df = NumBinner(columns_to_bin, cat_cols, num_cols).fit_transform(df)`
    """

    def __init__(self, columns_to_bin, cat_columns, num_columns):
        """NumBinner __init__.
        Args:
            columns_to_bin (list): list of numerical columns names in which data should be binned.
            cat_columns (list): list of categorical columns names.
            num_columns (list): list of numerical columns names.
        """
        self.columns_to_bin = columns_to_bin
        self.cat_columns = cat_columns
        self.num_columns = num_columns


    def transform(self, data):
        """Bin data in specified columns.
        Args:
            data (pd.DataFrame): DataFrame containing data for binning.
        Returns:
            data (pd.DataFrame): DataFrame with binned columns.
        """
        # Iterate through columns for binning.
        for col in self.columns_to_bin:

            # Generate bins using quantiles from 0 to 1 with step .25
            bins = [data[col].quantile(x/100) for x in range(0, 101, 25)]

            # Bin columns
            data[col] = pd.cut(data[col].values, bins)

            # Change column type and move it from numerical columns list to categorical columns list
            data[col] = data[col].astype('object')
            self.cat_columns.append(col)
            self.num_columns.remove(col)

        return self.cat_columns, self.num_columns, data


    def fit(self, *_):
        """Fit the transformer."""
        return self


class NumClipper(BaseEstimator, TransformerMixin):
    """Clipping data in specified numerical columns using low_quantile and high_quantile (params) as min and max values.
    Usage example:
        `df = NumClipper(columns_to_clip, low_q=.3, high_q=.97).fit_transform(df)`
    """

    def __init__(self, columns_to_clip, low_q=0.03, hight_q=0.97):
        """NumClipper __init__,
        Args:
            columns_to_clip (list): list of numerical columns names in which data should be clipped.
            low_q (float): low percentel specifies minimum value for clipping.
            high_q (float): high percentel specifies maximum value for clipping.
        """
        self.columns_to_clip = columns_to_clip
        self.low_q = low_q
        self.hight_q = hight_q


    def transform(self, data):
        """Clip data in spicified columns.
        Args:
            data (pd.DataFrame): DataFrame containing data for binning.
        Returns:
            data (pd.DataFrame): DataFrame with clipped data.
        """
        # Iterate  through columns to clip
        for col in self.columns_to_clip:

            # Define quantile values
            low_val = data[col].quantile(self.low_q)
            hight_val = data[col].quantile(self.hight_q)

            # Clip data
            data[[col]] = data[[col]].clip(low_val, hight_val)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self



class MyPandasQuantileTransformer(BaseEstimator, TransformerMixin):
    """Transform all numeric, non-categorical variables according to
    QuantileTransformer. Return pandas DataFrame"""

    def __init__(self, n_quantiles=100, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_dist = output_distribution

    def fit(self, X, y=None):
        self.transformer = QuantileTransformer(n_quantiles=self.n_quantiles,
                                               output_distribution=self.output_dist)
        self.cont_cols = get_cont_cols(X)
        self.transformer.fit(X[self.cont_cols])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.cont_cols] = self.transformer.transform(X[self.cont_cols])
        return X


class MyPandasStandardScaler(BaseEstimator, TransformerMixin):
    """Transform all numeric, non-categorical variables according to
    StandardScaler. Return pandas DataFrame"""

    def __init__(self, **kwargs):
        self.scaler = StandardScaler(**kwargs)

    def fit(self, X, y=None):
        self.cont_cols = get_cont_cols(X)
        self.scaler.fit(X[self.cont_cols])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.cont_cols] = self.scaler.transform(X[self.cont_cols])
        return X


class CutTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, bins, right = True, labels = None, include_lowest = True):
        self.bins = bins
        self.right = right
        self.labels = labels
        self.include_lowest = include_lowest

    def fit(self, X, y = None):
        X = column_or_1d(X, warn = True)
        return self

    def transform(self, X):
        X = column_or_1d(X, warn = True)
        return pd.cut(X, bins = self.bins, right = self.right, labels = self.labels, include_lowest = self.include_lowest)


class PowerFunctionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, power):
        if not isinstance(power, int):
            raise ValueError("Power {0} is not an integer".format(power))
        self.power = power

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return np.power(X, self.power)


class Scaler(BaseEstimator, TransformerMixin):
    """Scaling data in specified numerical columns with specified method
    Usage example:
        `scaled_df = Scaler(columns_to_scale, method='minmax').fit_transform(df)`
    """
    def __init__(self, columns_to_scale, method='standart_scaler'):
        """Scaler __init__.
        Args:
            columns_to_scale (list): list containing names of columns to scale.
            method (str): specifies name of scaling method (default 'standart_scaler')
            ['standart_scaler', 'minmax', 'robust'].
        """
        self.methods = ['standart_scaler', 'minmax', 'robust']
        self.columns_to_scale = columns_to_scale
        self.method = method

        # Raise assertion if passed method of scaling is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Scaling specified features.
        Args:
            data (pd.DataFrame): DataFrame containing data for scaling.
        Returns:
            data (pd.DataFrame): DataFrame with scaled numerical columns.
        """
        if self.method == 'standart_scaler':
            for col in self.columns_to_scale:
                data[col] = StandardScaler().fit_transform(data[[col]].values)

        elif self.method == 'minmax':
            for col in self.columns_to_scale:
                data[col] = MinMaxScaler().fit_transform(data[[col]].values)

        elif self.method == 'robust':
            for col in self.columns_to_scale:
                data[col] = RobustScaler().fit_transform(data[[col]].values)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self


class SkewRemover(BaseEstimator, TransformerMixin):
    """Removing skew from data in specified numerical columns with specified method.
    Usage example:
        `nonskew_df = SkewRemover(columns_to_remove_skew, lam=.15, skew_thresh=.75, method='boxcox').fit_transform(df)`
    """
    def __init__(self, columns_to_remove_skewn, lam=.15, skew_thresh=.75, method='log'):
        """SkewRemover __init__
        Args:
            columns_to_remove_skewn (list): list containing names of columns to unskew.
            lam (float): BoxCox transformation parameter.
            skew_thresh (float): threshold for detecting skewed columns.
            method (str): specifies name of transformation method (default 'log')
            ['log', 'boxcox'].
        """
        self.methods = ['log', 'boxcox']
        self.method = method
        self.cols = columns_to_remove_skewn
        self.thresh = skew_thresh
        self.lam = lam

        # Raise assertion if passed method of transformation is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Remove skew from data.
        Args:
            data (pd.DataFrame): DataFrame containing data for transformation.
        Returns:
            data (pd.DataFrame): DataFrame with transfored data.
        """
        # Compute skew of each feature
        skewed_feats = data[self.cols].apply(lambda x: skew(x))

        # Filter features to transform by skew threshold
        skewed_feats = skewed_feats[abs(skewed_feats) > self.thresh]

        if self.method == 'log':
            for feat in skewed_feats.index:
                data[feat] = np.log1p(data[feat])

        elif self.method == 'boxcox':
            for feat in skewed_feats.index:
                data[feat] = boxcox1p(data[feat], self.lam)

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self

class NumNanFiller(BaseEstimator, TransformerMixin):
    """Filling NaNs in numerical columns with mean, median or custom value."""

    def __init__(self, num_cols, method='mean'):
        """NumNanFiller __init__.
        Args:
            num_cols (list): ;ist of numerical columns names.
            method (str or float or int): specifies how to fill NaNs (default 'mean')
            ['mean', 'median', int, float].
        """
        # Define known methods list
        self.methods = ['mean', 'median']
        self.method = method
        self.num_cols = num_cols

        # Raise assertion if passed method of filling NaNs is unknow
        assert (self.method in self.methods) or (type(self.method) in [float, int]), f'Unknow method(use number or one of {self.methods})'


    def transform(self, data):
        """Fill NaNs in numerical columns.
        Args:
            data (pd.DataFrame): DataFrame to fill.
        Returns:
            data (pd.DataFrame): DataFrame with filled NaNs in numerical columns.
        """
        if self.method == 'mean':
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].mean())

        elif self.method == 'median':
            for col in self.num_cols:
                data[col] = data[col].fillna(data[col].median())

        else:
            for col in self.num_cols:
                data[col] = data[col].fillna(self.method)

        return data

    def fit(self, *_):
        """Fit the transformer."""
        return self

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class DFRobustScaler(TransformerMixin):
    # RobustScaler but for pandas DataFrames

    def __init__(self):
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        self.rs.fit(X)
        self.center_ = pd.Series(self.rs.center_, index=X.columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xrs = self.rs.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled





class ClipTransformer(TransformerMixin):

    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xclip = np.clip(X, self.a_min, self.a_max)
        return Xclip



class BoxCox(BaseEstimator, TransformerMixin):
    """Perform BoxCox transformation on continuous numeric data."""

    # TODO: Remove this internal function and use PowerTransform from sklearn
    # when sklearn version is upgraded to 0.20

    def fit(self, X, y=None):
        """Fit translate and lambda attributes to X data.
        Args:
            X (:obj:`np.ndarray`): Fit data
        Returns:
            self
        """
        X = check_array(X)
        min_ = np.nanmin(X)
        self.translate_ = -min_ if min_ <= 0 else 0
        _, self.lambda_ = boxcox(X + 1 + self.translate_)
        return self

    def transform(self, X):
        """Perform Box Cox transform on input.
        Args:
            X (:obj:`np.ndarray`): X data
        Returns:
            :obj:`np.ndarray`: Transformed data
        """
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = boxcox(X + 1 + self.translate_, self.lambda_)
        return X

    def inverse_transform(self, X):
        """Reverse Box Cox transform.
        Args:
            X (:obj:`np.ndarray`): Transformed X data
        Returns:
            :obj:`np.ndarray`: Original data
        """
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = np.clip(X, a_min=-0.99, a_max=None)
        X = inv_boxcox1p(X, self.lambda_) - self.translate_
        return X


