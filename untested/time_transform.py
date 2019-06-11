class DurationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, year):
        if year < 1900:
            raise ValueError("Year {0} is earlier than 1900".format(year))
        self.epoch = datetime(year, 1, 1, tzinfo = None)

    def _to_duration(self, td):
        return td

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        shape = X.shape
        if len(shape) > 1:
            X = X.ravel()
        Xt = pd.to_timedelta(X - self.epoch)
        Xt = self._to_duration(Xt)
        if len(shape) > 1:
            Xt = Xt.reshape(shape)
        return Xt


class DaysSinceYearTransformer(DurationTransformer):

    def __init__(self, year):
        super(DaysSinceYearTransformer, self).__init__(year)

    def _to_duration(self, td):
        return (td.days).values


class SecondsSinceYearTransformer(DurationTransformer):

    def __init__(self, year):
        super(SecondsSinceYearTransformer, self).__init__(year)

    def _to_duration(self, td):
        return ((td.total_seconds()).values).astype(int)


class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class HourOfDayTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        hours = DataFrame(X['datetime'].apply(lambda x: x.hour))
        return hours

    def fit(self, X, y=None, **fit_params):
        return self


