class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers

    Fit several DataFrame transformers and provides a concatenated
    Data Frame

    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers

    """

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        concatted :  pandas DataFrame

        """

        concatted = pd.concat([transformer.transform(X)
                               for transformer in
                               self.fitted_transformers_], axis=1).copy()
        return concatted

    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self : object
        """

        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    Parameters
    ----------
    impute : Boolean, default False
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]
    """

    def __init__(self, func, impute=False):
        self.func = func
        self.impute = impute
        self.series = pd.Series()

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

        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
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

        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).copy()
        return self


class ColumnSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection

    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.

    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: []

    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
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

    def transform(self, X):
        """ Selects columns of a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------

        X : pandas DataFrame
            contains selected columns of X
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)



class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers

    Fit several DataFrame transformers and provides a concatenated
    Data Frame

    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers

    """

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        concatted :  pandas DataFrame

        """

        concatted = pd.concat([transformer.transform(X)
                               for transformer in
                               self.fitted_transformers_], axis=1).copy()
        return concatted

    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self : object
        """

        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self



class ColumnSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection

    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.

    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: []

    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
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

    def transform(self, X):
        """ Selects columns of a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------

        X : pandas DataFrame
            contains selected columns of X
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class DFFunctionTransformer(TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt


class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))


class ApplyFunction(BaseEstimator, TransformerMixin):

    def __init__(self, columns, fun):
        self.columns = columns
        self.fun = fun

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for col in self.columns:
            x[col] = x[col].map(self.fun)
        return x


