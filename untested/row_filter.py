

class ExistFeaturesTransform(BaseEstimator, TransformerMixin):
    """Filter rows based on whether a specified set of features exists
    Parameters
    ----------
    included : list str
       list of features that need to exist
    """

    def __init__(self, included=None):
        super(ExistFeaturesTransform, self).__init__()
        self.included = included

    def fit(self, objs):
        return self

    def transform(self, df):
        """
        Transform by returning input feature set if required features exist in it
        Parameters
        ----------
        df : pandas dataframe
        Returns
        -------

        Transformed pandas dataframe
        """
        df.dropna(subset=self.included, inplace=True)
        return df


class DataframeNullValueFilter(TransformerMixin):
    """Given a pandas dataframe, remove rows that contain null values in any column except the excluded."""

    def __init__(self, excluded_columns=None):
        # TODO validate excluded column is a list
        self.excluded_columns = excluded_columns or []

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        subset = [c for c in x.columns if c not in self.excluded_columns]

        x.dropna(axis=0, how='any', inplace=True, subset=subset)

        if x.empty:
            raise TypeError(
                "Because imputation is set to False, rows with missing or null/NaN values are being dropped. "
                "In this case, all rows contain null values and therefore were ALL dropped. "
                "Please consider using imputation or assessing the data quality and availability")

        return x