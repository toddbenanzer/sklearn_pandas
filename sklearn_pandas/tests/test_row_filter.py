from unittest import TestCase
from sklearn_pandas.transformers.row_filter import *
import pandas as pd


class TestDropNARowFilter(TestCase):

    def test_no_missing(self):
        X = pd.DataFrame({'A': [1, 2, 3]})
        filter = DropNARowFilter()
        pd.testing.assert_frame_equal(filter.fit_transform(X), X)

    def test_nan(self):
        X = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        filter = DropNARowFilter()
        pd.testing.assert_frame_equal(filter.fit_transform(X), pd.DataFrame({'A': [1.0, 3.0]}, index=[0, 2]))

    def test_inf(self):
        X = pd.DataFrame({'A': [1.0, np.inf, 3.0]})
        filter = DropNARowFilter()
        pd.testing.assert_frame_equal(filter.fit_transform(X), pd.DataFrame({'A': [1.0, 3.0]}, index=[0, 2]))