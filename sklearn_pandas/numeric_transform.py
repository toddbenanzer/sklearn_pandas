import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn_pandas.util import retain_sign


class QuantileBinning(BaseEstimator, TransformerMixin):

    def __init__(self, nbins=5):
        self.nbins = nbins

    def fit(self, X, y=None, **fitparams):
        self.cuts = {}
        for col in X.columns:
            self.cuts[col] = X[col].quantile(q=np.linspace(0, 1, nbins+1))
        return self

    def transform(self, X, **transformparams):
        for col in X.columns:
            X[col] = pd.cut(x=X[col], bins=self.cuts[col])
        return X


