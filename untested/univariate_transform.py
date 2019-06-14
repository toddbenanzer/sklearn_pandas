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
