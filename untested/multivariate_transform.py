

class MyPandasPCA(BaseEstimator, TransformerMixin):
    """Transform `X` according to PCA. Return pandas DataFrame"""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.input_columns = np.array(X.columns)
        self.transformer = PCA(n_components=self.n_components, svd_solver='full')
        self.transformer.fit(X[self.input_columns], y)
        self.n_cols = len(self.transformer.components_)
        self.output_columns = [f'pca{ii + 1:02}' for ii in range(self.n_cols)]
        return self

    def transform(self, X, y=None):
        X_trans = self.transformer.transform(X[self.input_columns])
        n_cols = X_trans.shape[1]
        df_out = pd.DataFrame(X_trans, columns=self.output_columns)
        return df_out


class MyPandasPolynomialFeatures(BaseEstimator, TransformerMixin):
    """Make polynomial interactions according to PolynomialFeatures.
    Returns pandas DataFrame"""

    def __init__(self, interaction_only=True, include_bias=False):
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.transformer = PolynomialFeatures(interaction_only=self.interaction_only,
                                              include_bias=self.include_bias)
        self._transformer = clone(self.transformer)
        self.input_columns = list(X.columns)
        self._transformer.fit(X)
        return self

    def transform(self, X, y=None):
        transformed_col_names = self._transformer.get_feature_names(self.input_columns)
        X_trans = self._transformer.transform(X)
        df_out = pd.DataFrame(X_trans, columns=transformed_col_names)
        return df_out


class SingleColInteractions(BaseEstimator, TransformerMixin):
    """Make interaction terms between column `col_constant`, and all the columns
    in `col_interactions`. Returns pandas DataFrame
    """

    def __init__(self, col_constant, cols_interactions):
        self.col_constant = col_constant
        self.cols_interactions = cols_interactions
        assert isinstance(cols_interactions, list)

        self.interaction_terms = []
        for col in self.cols_interactions:
            interaction_name = f'{self.col_constant} x {col}'
            self.interaction_terms.append(interaction_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        assert isinstance(X, pd.DataFrame)
        for col in self.cols_interactions:
            interaction_name = f'{self.col_constant} x {col}'
            X[interaction_name] = X[self.col_constant] * X[col]
        return X

    def get_feature_names(self):
        return self.interaction_terms



class Aggregator(BaseEstimator, TransformerMixin):

    def __init__(self, function):
        functions = ["min", "max", "mean"]
        if function not in functions:
            raise ValueError("Function {0} not in {1}".format(function, functions))
        self.function = function

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.function == "min":
            return np.amin(X, axis = 1)
        elif self.function == "max":
            return np.amax(X, axis = 1)
        elif self.function == "mean":
            return np.mean(X, axis = 1)
        return X


class group_by_featurizer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col, feature):
        self.group_col = group_col
        self.value_col = value_col
        self.feature = feature
        self.gb = None

    def fit(self, X):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(self.group_col, list)
        self.gb = X.groupby(self.group_col, as_index=False).agg({self.value_col:self.feature})
        return self

    def transform(self, X):
        self.fit(X)
        return pd.merge(X, self.gb, on=self.group_col)
