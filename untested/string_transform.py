class ConcatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, separator = ""):
        self.separator = separator

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        func = lambda x: self.separator.join([str(v) for v in x])
        Xt = eval_rows(X, func)
        if isinstance(Xt, Series):
            Xt = Xt.values
        return Xt.reshape(-1, 1)


class ExpressionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, expr):
        self.expr = expr

    def _eval_row(self, X):
        return eval(self.expr)

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        func = lambda x: self._eval_row(x)
        Xt = eval_rows(X, func)
        if isinstance(Xt, Series):
            Xt = Xt.values
        return Xt.reshape(-1, 1)


class LookupTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, mapping, default_value):
        if type(mapping) is not dict:
            raise ValueError("Input value to output value mapping is not a dict")
        for k, v in mapping.items():
            if k is None:
                raise ValueError("Key is None")
        self.mapping = mapping
        self.default_value = default_value

    def _transform_dict(self):
        transform_dict = defaultdict(lambda: self.default_value)
        transform_dict.update(self.mapping)
        return transform_dict

    def fit(self, X, y = None):
        X = column_or_1d(X, warn = True)
        return self

    def transform(self, X):
        X = column_or_1d(X, warn = True)
        transform_dict = self._transform_dict()
        func = lambda k: transform_dict[k]
        if hasattr(X, "apply"):
            return X.apply(func)
        return np.vectorize(func)(X)


def _regex_engine(pattern):
    try:
        import pcre
        return pcre.compile(pattern)
    except ImportError:
        warnings.warn(
            "Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns",
            Warning)
        import re
        return re.compile(pattern)


class ReplaceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pattern, replacement):
        self.pattern = pattern
        self.replacement = replacement

    def fit(self, X, y=None):
        X = column_or_1d(X, warn=True)
        return self

    def transform(self, X):
        X = column_or_1d(X, warn=True)
        engine = _regex_engine(self.pattern)
        func = lambda x: engine.sub(self.replacement, x)
        return eval_rows(X, func)


class SubstringTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, begin, end):
        if begin < 0:
            raise ValueError("Begin position {0} is negative".format(begin))
        if end < begin:
            raise ValueError("End position {0} is smaller than begin position {1}".format(end, begin))
        self.begin = begin
        self.end = end

    def fit(self, X, y=None):
        X = column_or_1d(X, warn=True)
        return self

    def transform(self, X):
        X = column_or_1d(X, warn=True)
        func = lambda x: x[self.begin:self.end]
        return eval_rows(X, func)


class StringNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, function=None, trim_blanks=True):
        functions = ["lowercase", "uppercase"]
        if (function is not None) and (function not in functions):
            raise ValueError("Function {0} not in {1}".format(function, functions))
        self.function = function
        self.trim_blanks = trim_blanks

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        Xt = X.astype("U")
        # Transform
        if self.function == "lowercase":
            Xt = np.char.lower(Xt)
        elif self.function == "uppercase":
            Xt = np.char.upper(Xt)
        # Trim blanks
        if self.trim_blanks:
            Xt = np.char.strip(Xt)
        return Xt


class MatchesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X, y = None):
        X = column_or_1d(X, warn = True)
        return self

    def transform(self, X):
        X = column_or_1d(X, warn = True)
        engine = _regex_engine(self.pattern)
        func = lambda x: bool(engine.search(x))
        return eval_rows(X, func)


class StringTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xstr = X.applymap(str)
        return Xstr



