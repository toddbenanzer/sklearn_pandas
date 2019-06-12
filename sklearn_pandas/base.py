import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def fit(self, X, y=None, **fitparams):
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self

    def transform(self, X, **transformparams):
        df_concat = pd.concat([t.transform(X) for t in self.fitted_transformers_], axis=1).copy()
        return df_concat


class DataFrameModelTransformer(TransformerMixin):

    def __init__(self, model, output_column_names):
        self.model = model
        self.output_column_names = output_column_names

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X), columns=self.output_column_names)
