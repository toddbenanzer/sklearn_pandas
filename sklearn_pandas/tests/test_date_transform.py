from unittest import TestCase
import datetime
from sklearn_pandas.date_transform import *


class TestDateTransform(TestCase):

    def test_easy_date(self):
        df = pd.DataFrame({'A': ['2019-03-01', ]})
        df_expected = pd.DataFrame({'A': [datetime.date(2019, 3, 1), ]}, dtype='datetime64[ns]')
        transform = DateTransform()
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)

    def test_easy_date2(self):
        df = pd.DataFrame({'A': ['2019/03/01', ]})
        df_expected = pd.DataFrame({'A': [datetime.date(2019, 3, 1), ]}, dtype='datetime64[ns]')
        transform = DateTransform()
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)

    def test_easy_date3(self):
        df = pd.DataFrame({'A': ['March 01, 2019', ]})
        df_expected = pd.DataFrame({'A': [datetime.date(2019, 3, 1), ]}, dtype='datetime64[ns]')
        transform = DateTransform()
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)

    def test_incomplete_date(self):
        df = pd.DataFrame({'A': ['March 01', ]})
        df_expected = pd.DataFrame({'A': [pd.NaT, ]}, dtype='datetime64[ns]')
        transform = DateTransform()
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


class TestExtractDatePart(TestCase):

    def test_simple_conversion(self):
        df = pd.DataFrame({'A': ['2019-03-01', ]})
        df_expected = pd.DataFrame({
            'A_year': [2019, ],
            'A_month': [3, ],
            'A_day': [1, ],
            'A_hour': [0, ],
            'A_minute': [0, ],
            'A_second': [0, ],
            'A_weekday': [4, ],
            'A_weekday_name': ['Friday', ],
            'A_quarter': [1, ],
            'A_dayofyear': [60, ],
            'A_weekofyear': [9, ],
        })
        transform = ExtractDatePart()
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)

    def test_subset_conversion(self):
        df = pd.DataFrame({'A': ['2019-03-01', ]})
        df_expected = pd.DataFrame({
            #'A_year': [2019, ],
            #'A_month': [3, ],
            'A_day': [1, ],
            'A_hour': [0, ],
            'A_minute': [0, ],
            'A_second': [0, ],
            'A_weekday': [4, ],
            'A_weekday_name': ['Friday', ],
            'A_quarter': [1, ],
            'A_dayofyear': [60, ],
            'A_weekofyear': [9, ],
        })
        transform = ExtractDatePart(get_year=False, get_month=False)
        pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)