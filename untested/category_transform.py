class OneHotEncodeTransform(BaseEstimator, TransformerMixin):
    """
    Encode all categorical features except those which are present in excluded parameter
    """

    def __init__(self, excluded=[]):
        self.excluded = excluded
        self.encoders_map = {}

    def fit(self, X, y=None):
        self.dvec = DictVectorizer(sparse=False)
        X = self.dvec.fit(X.transpose().to_dict().values())
        return self

    def transform(self, X):
        X_array_encoded = self.dvec.transform(X.transpose().to_dict().values())
        res = pd.DataFrame(X_array_encoded, columns=self.dvec.get_feature_names())
        return res


class CatDummifier(BaseEstimator, TransformerMixin):
    """Generating dummy features for specified categorical columns.
    Usage example:
        `dummified_df = CatDummifier(cat_cols, drop_first=True).fit_transform(df)`
    """

    def __init__(self, cat_columns, drop_first=True):
        """CatDummifier __init__.
        Args:
            cat_columns (list): list of categorical columns names.
            drop_first (bool): if true - generate k-1 one-hot columns from k categories.
        """
        self.cat_columns = cat_columns
        self.drop_first = drop_first

    def transform(self, data):
        """Dummifier categorical columns.
        Args:
            data (pd.DataFrame): DataFrame containing data for generating dummies.
        Returns:
            data (pd.DataFrame): DataFrame with dummified categorical columns.
        """
        if self.drop_first:
            return pd.get_dummies(data, columns=self.cat_columns, drop_first=self.drop_first)
        else:
            return pd.get_dummies(data, columns=self.cat_columns)

    def fit(self, *_):
        """Fit the transformer."""
        return self


class CatLabelEncoder(BaseEstimator, TransformerMixin):
    """Encoding categorical data in specified columns by mapping each category to number
    Usage example:
        `le_df = CatLabelEncoder(cat_cols).fit_transform(df)`
    """

    def __init__(self, cat_columns):
        """CatLabelEncoder __init__.
        Args:
            cat_columns (list): list of categorical columns names.
        """
        self.cat_columns = cat_columns


    def transform(self, data):
        """Encode categorical columns.
        Args:
            data (pd.DataFrame): DataFrame containing data for encoding.
        Returns:
            data (pd.DataFrame): DataFrame with encoded categorical columns.
        """
        # Iterate  through columns to encode
        for col in self.cat_columns:
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes

        return data

    def fit(self, *_):
        return self




class CatNanFiller(BaseEstimator, TransformerMixin):
    """Filling NaNs in categorical columns with most frequent value of special indicator value."""

    def __init__(self, cat_cols, method='top'):
        """CatNanFiller __init__.
        Args:
            cat_cols (list): list of categorical columns names.
            method (str): specifies how to fill NaNs (default 'top')
            ['top' - fill with more frequent, 'indicator' - fill with special value].
        """
        # Define known methods list
        self.methods = ['top', 'indicator']
        self.method = method
        self.cat_cols = cat_cols

        # Raise assertion if passed method of filling NaNs is unknow
        assert self.method in self.methods, f'Unknow method(use one of {self.methods})'


    def transform(self, data):
        """Fill NaNs in categorical columns.
        Args:
            data (pd.DataFrame): DataFrame to fill.
        Returns:
            data (pd.DataFrame): DataFrame with filled NaNs in categorical columns.
        """
        if self.method == 'top':
            for col in self.cat_cols:
                data[col] = data[col].fillna(data[col].describe()['top'])

        elif self.method == 'indicator':
            for col in self.cat_cols:
                data[col] = data[col].fillna('NO_VALUE')

        return data


    def fit(self, *_):
        """Fit the transformer."""
        return self



class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = self.prep(X)
        unique_vals = []
        for column in X.T:
            unique_vals.append(np.unique(column))
        self.unique_vals = unique_vals

    def transform(self, X, y=None):
        X = self.prep(X)
        unique_vals = self.unique_vals
        new_columns = []
        for i, column in enumerate(X.T):
            num_uniq_vals = len(unique_vals[i])
            encoder_ring = dict(zip(unique_vals[i], range(len(unique_vals[i]))))
            f = lambda val: encoder_ring[val]
            f = np.vectorize(f, otypes=[np.int])
            new_column = np.array([f(column)])
            if num_uniq_vals <= 2:
                new_columns.append(new_column)
            else:
                one_hots = np.zeros([num_uniq_vals, len(column)], np.int)
                one_hots[new_column, range(len(column))] = 1
                new_columns.append(one_hots)
        new_columns = np.concatenate(new_columns, axis=0).T
        return one_hots

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def prep(X):
        shape = X.shape
        if len(shape) == 1:
            X = X.values.reshape(shape[0], 1)
        return X


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=1, dummy_na=True):
        self.min_frequency = min_frequency
        self.dummy_na = dummy_na
        self.categories = dict()
        self.features = []

    def fit(self, X):
        for col in X.columns:
            counts = pd.value_counts(X[col])
            self.categories[col] = list(set(counts[counts >= self.min_frequency].index.tolist()))
        return self

    def transform(self, X, *_):
        for col in X.columns:
            X = X.astype({col: CategoricalDtype(self.categories[col], ordered=True)})
        ret = pd.get_dummies(X, dummy_na=self.dummy_na)
        self.features = ret.columns
        return ret

    def get_feature_names(self):
        return self.features


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'CategoricalImputer':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X



class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise TypeError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X


class UncommonRemover(BaseEstimator, TransformerMixin):
    """Merge uncommon values in a categorical column to an other value.
    Note: Unseen values from fitting will also be merged.
    Args:
        threshold (float): data that is less frequent than this percentage
            will be merged into a singular unique value
        replacement (Optional): value with which to replace uncommon values
    """

    def __init__(self, threshold=0.01, replacement="UncommonRemover_Other"):
        self.threshold = threshold
        self.replacement = replacement

    def fit(self, X, y=None):
        """Find the uncommon values and set the replacement value.
        Args:
            X (:obj:`pd.DataFrame`): input dataframe
        Returns:
            self
        """
        X = check_df(X, single_column=True).iloc[:, 0]

        vc_series = X.value_counts()
        self.values_ = vc_series.index.values.tolist()
        self.merge_values_ = vc_series[
            vc_series <= (self.threshold * X.size)
        ].index.values.tolist()

        return self

    def transform(self, X, y=None):
        """Apply the computed transform to the passed in data.
        Args:
            X (:obj:`pd.DataFrame`): input DataFrame
        Returns:
            :obj:`pd.DataFrame`: transformed dataframe
        """
        X = check_df(X, single_column=True).iloc[:, 0]
        check_is_fitted(self, ["values_", "merge_values_"])
        X[
            X.isin(self.merge_values_) | ~X.isin(self.values_)
        ] = self.replacement
        X = X.to_frame()

        return X



