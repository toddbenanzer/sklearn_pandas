

class DataframeColumnSuffixFilter(TransformerMixin):
    """Given a pandas dataframe, remove columns with suffix 'DTS'."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in x.columns if not column.endswith('DTS')]

        # Select all data excluding datetime columns
        return x[filtered_column_names]


class DataFrameColumnDateTimeFilter(TransformerMixin):
    """Given a pandas dataframe, remove any columns that has the type datetime."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Select all data excluding datetime columns
        return x.select_dtypes(exclude=["datetime"])


class DataframeColumnRemover(TransformerMixin):
    """Given a pandas dataframe, remove the given column or columns in list form."""

    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove

    def fit(self, x, y=None):
        return self

    def transform(self, X, y=None):
        validate_dataframe_input(X)
        if self.columns_to_remove is None:
            # if there is no grain column, for example
            return X

        # Build a list of all columns except for the grain column'
        filtered_column_names = [c for c in X.columns if c not in self.columns_to_remove]

        # return the filtered dataframe
        return X[filtered_column_names]


class IncludeFeaturesTransform(BaseEstimator, TransformerMixin):
    """
    Filter a dataset and include only specided set of features
    Parameters
    ----------

    input_features : list str
       input features to include
    """

    def __init__(self, included=[]):
        self.included = included

    def fit(self, X):
        """nothing to do in fit
        """
        return self

    def transform(self, df):
        """
        transform a dataframe to include given features
        Parameters
        ----------
        df : pandas dataframe
        Returns
        -------

        Transformed pandas dataframe
        """
        df = df[list(set(self.included).intersection(df.columns))]
        return df

class ExcludeFeaturesTransform(BaseEstimator, TransformerMixin):
    """
    Filter a dataset and exclude specided set of features
    Parameters
    ----------
    excluded : list str
       list of features to be excluded
    """

    def __init__(self, excluded=[]):
        self.excluded = excluded

    def fit(self, X):
        """nothing to do in fit
        """
        return self

    def transform(self, df):
        """
        Trasform dataframe to include specified features only
        Parameters
        ----------
        df : pandas dataframe
        Returns
        -------

        Transformed pandas dataframe
        """
        df = df.drop(self.excluded, axis=1, errors='ignore')
        return df



class MyPandasVarianceThreshold(BaseEstimator, TransformerMixin):
    """Select columns with variance above `threshold`. Return pandas DataFrame"""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.transformer = VarianceThreshold(threshold=self.threshold)
        self.transformer.fit(X, y)
        self.input_columns = np.array(X.columns)
        support = self.transformer.get_support()
        self.output_columns = self.input_columns[support]
        return self

    def transform(self, X, y=None):
        X_trans = self.transformer.transform(X)
        df_out = pd.DataFrame(X_trans, columns=self.output_columns)
        return df_out


class TrainTestObjectTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    Parameters
    ----------
    impute : Boolean, default False
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]
    """

    def __init__(self, test):
        self.test = test

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X
        """

        object_levels = np.union1d(X.fillna('NAN'), self.test.fillna('NAN'))
        trans = pd.DataFrame(X).apply(lambda x: x.astype('category', categories=object_levels)).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self
        """

        return self


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides dropping all-zero columns
    """

    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transformparams):
        """ Drops certain all-zero columns of X

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : DataFrame
        """

        trans = X.drop(self.columns, axis=1).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Determines the all-zero columns of X

        Parameters
        ----------
        X : DataFrame
        y : not used

        Returns
        ----------
        self : object
        """
        return self

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection

    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.

    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: []

    """

    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------

        trans : pandas DataFrame
            contains selected columns of X
        """
        trans = X[self.columns].copy()
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function

        Parameters
        ----------
        X : pandas DataFrame
        y : default None


        Returns
        ----------
        self
        """
        return self


class DataframeColumnSuffixFilter(TransformerMixin):
    """Given a pandas dataframe, remove columns with suffix 'DTS'."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in x.columns if not column.endswith('DTS')]

        # Select all data excluding datetime columns
        return x[filtered_column_names]


