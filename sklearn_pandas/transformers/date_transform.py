import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.util import validate_columns_exist, validate_dataframe


class DateTransform(BaseEstimator, TransformerMixin):

    def __init__(self, errors='coerce', dayfirst=False, yearfirst=False, utc=None, format=None, exact=False):
        self.errors = errors
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc
        self.format = format
        self.exact = exact

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        return self

    def _to_datetime(self, x):
        return pd.to_datetime(x, errors=self.errors, dayfirst=self.dayfirst, yearfirst=self.yearfirst, utc=self.utc,
                              format=self.format, exact=self.exact, unit=None, infer_datetime_format=False,
                              origin='unix', cache=False)

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        # assumes X is a DataFrame
        Xdate = X.apply(self._to_datetime)
        return Xdate


class ExtractDatePart(BaseEstimator, TransformerMixin):

    def __init__(self, get_year=True, get_month=True, get_day=True, get_hour=True, get_minute=True, get_second=True,
                 get_weekday=True, get_weekday_name=True, get_quarter=True, get_dayofyear=True, get_weekofyear=True,
                 delim='_', errors='coerce', dayfirst=False, yearfirst=False, utc=None, format=None, exact=False):
        self.get_year = get_year
        self.get_month = get_month
        self.get_day = get_day
        self.get_hour = get_hour
        self.get_minute = get_minute
        self.get_second = get_second
        self.get_weekday = get_weekday
        self.get_weekday_name = get_weekday_name
        self.get_quarter = get_quarter
        self.get_dayofyear = get_dayofyear
        self.get_weekofyear = get_weekofyear
        self.delim = delim
        self.errors = errors
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.utc = utc
        self.format = format
        self.exact = exact

    def fit(self, X, y=None, **fitparams):
        X = validate_dataframe(X)
        return self

    def _to_datetime(self, x):
        return pd.to_datetime(x, errors=self.errors, dayfirst=self.dayfirst, yearfirst=self.yearfirst, utc=self.utc,
                              format=self.format, exact=self.exact, unit=None, infer_datetime_format=False,
                              origin='unix', cache=False)

    def transform(self, X, **transformparams):
        X = validate_dataframe(X)
        X = X.copy()
        new_col_list = []
        for col in X.columns:
            if self.get_year:
                new_col = col + self.delim + 'year'
                X[new_col] = self._to_datetime(X[col]).dt.year
                new_col_list.append(new_col)
            if self.get_month:
                new_col = col + self.delim + 'month'
                X[new_col] = self._to_datetime(X[col]).dt.month
                new_col_list.append(new_col)
            if self.get_day:
                new_col = col + self.delim + 'day'
                X[new_col] = self._to_datetime(X[col]).dt.day
                new_col_list.append(new_col)
            if self.get_hour:
                new_col = col + self.delim + 'hour'
                X[new_col] = self._to_datetime(X[col]).dt.hour
                new_col_list.append(new_col)
            if self.get_minute:
                new_col = col + self.delim + 'minute'
                X[new_col] = self._to_datetime(X[col]).dt.minute
                new_col_list.append(new_col)
            if self.get_second:
                new_col = col + self.delim + 'second'
                X[new_col] = self._to_datetime(X[col]).dt.second
                new_col_list.append(new_col)
            if self.get_weekday:
                new_col = col + self.delim + 'weekday'
                X[new_col] = self._to_datetime(X[col]).dt.weekday
                new_col_list.append(new_col)
            if self.get_weekday_name:
                new_col = col + self.delim + 'weekday_name'
                #X[new_col] = self._to_datetime(X[col]).dt.weekday_name
                X[new_col] = self._to_datetime(X[col]).dt.day_name()
                new_col_list.append(new_col)
            if self.get_quarter:
                new_col = col + self.delim + 'quarter'
                X[new_col] = self._to_datetime(X[col]).dt.quarter
                new_col_list.append(new_col)
            if self.get_dayofyear:
                new_col = col + self.delim + 'dayofyear'
                X[new_col] = self._to_datetime(X[col]).dt.dayofyear
                new_col_list.append(new_col)
            if self.get_weekofyear:
                new_col = col + self.delim + 'weekofyear'
                X[new_col] = self._to_datetime(X[col]).dt.weekofyear
                new_col_list.append(new_col)

        return X.loc[:, new_col_list]
