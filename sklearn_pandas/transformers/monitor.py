import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn_pandas.util import validate_dataframe


class MonitorMixin(object):

    def print_message(self, message):
        if self.logfile:
            with open(self.logfile, "a") as fout:
                fout.write(message)
        else:
            print(message)


class ValidateTypes(BaseEstimator, TransformerMixin, MonitorMixin):

    def __init__(self, logfile=None, to_screen=True):
        self.logfile = logfile
        self.to_screen = to_screen

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.types = {}
        for col in X.columns:
            self.types[col] = X[col].dtype.name
        return self

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        new_col_list = []
        for col in X.columns:
            var_type = X[col].dtype.name
            if var_type != self.types[col]:
                self.print_message(
                    'Data Type Mismatch for column {col}: Expected {expected} Received {received}'.format(
                        col=col, expected=self.types[col], received=var_type)
                )

        return X


class ValidateRange(BaseEstimator, TransformerMixin, MonitorMixin):

    def __init__(self, logfile=None, to_screen=True, max_nunique=20):
        self.logfile = logfile
        self.to_screen = to_screen
        self.max_nunique = max_nunique

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        self.types = {}
        self.unique_vals = {}
        self.minmax = {}
        for col in X.columns:
            self.types[col] = X[col].dtype.name
            if self.types[col] in ('object', 'bool', 'category'):
                unique_values = X[col].unique()
                if len(unique_values) <= self.max_nunique:
                    self.unique_vals[col] = unique_values
                else:
                    self.unique_vals[col] = None
            elif self.types[col] in ('int64', 'float64', 'datetime64', 'timedelta'):
                self.minmax[col] = (X[col].min(), X[col].max())
        return self


    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        new_col_list = []
        for col in X.columns:
            var_type = X[col].dtype.name
            if self.types[col] in ('object', 'bool', 'category'):
                if self.unique_vals[col] is not None:
                    not_in_list = ~X[col].isin(self.unique_vals[col])
                    if sum(not_in_list) > 0:
                        new_values = str(X[col][not_in_list].unique().tolist())
                        self.print_message(
                            'New Categories specified for column {col}: Received {received}'.format(
                                col=col, received=new_values)
                        )
            elif self.types[col] in ('int64', 'float64', 'datetime64', 'timedelta'):
                minX = X[col].min()
                maxX = X[col].max()
                if minX < self.minmax[col][0]:
                    self.print_message(
                        'Low Value warning for column {col}: Lowest Training value {lowtrain}, Lowest Scoring value {lowscore}'.format(
                            col=col, lowtrain=self.minmax[col][0], lowscore=minX)
                    )
                if maxX > self.minmax[col][1]:
                    self.print_message(
                        'High Value warning for column {col}: Largest Training value {hightrain}, Largest Scoring value {highscore}'.format(
                            col=col, hightrain=self.minmax[col][1], highscore=maxX)
                    )

        return X

