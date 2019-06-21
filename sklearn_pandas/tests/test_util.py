from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn_pandas.util import is_pandas_dataframe, validate_dataframe, validate_columns_exist, retain_sign


class TestIs_Dataframe(TestCase):

    def test_None(self):
        self.assertFalse(is_pandas_dataframe(None))

    def test_dataframe(self):
        df = pd.DataFrame()
        self.assertTrue(is_pandas_dataframe(df))

    def test_not_dataframe(self):
        df = [[1, 4, 5], [-5, 8, 9]]
        self.assertFalse(is_pandas_dataframe(df))

        df = np.array([[1, 4, 5], [-5, 8, 9]])
        self.assertFalse(is_pandas_dataframe(df))


class TestValidate_dataframe(TestCase):

    def test_None(self):
        self.assertRaises(TypeError, validate_dataframe, None)

    def test_dataframe(self):
        df = pd.DataFrame()
        validate_dataframe(df)

    def test_not_dataframe(self):
        df = [[1, 4, 5], [-5, 8, 9]]
        self.assertRaises(TypeError, validate_dataframe, df)

        df = ([1, 4, 5], [-5, 8, 9])
        self.assertRaises(TypeError, validate_dataframe, df)


class Testvalidate_columns_exist(TestCase):

    def test_None(self):
        df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
        self.assertRaises(TypeError, validate_columns_exist, df, None)

    def test_column_doesnt_exist(self):
        df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
        self.assertRaises(KeyError, validate_columns_exist, df, ['C'])

    def test_column_does_exist(self):
        df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
        validate_columns_exist(df, ['A'])


class TestRetain_sign(TestCase):

    def test_single_value(self):
        self.assertEqual(retain_sign(np.sqrt)(-4), -2.0)

    def test_array_apply(self):
        df = pd.DataFrame({'A': [-1, -4, -9, ]})
        result = df['A'].apply(retain_sign(np.sqrt))
        expected_result = pd.DataFrame({'A': [-1.0, -2.0, -3.0, ], })
        pd.testing.assert_series_equal(expected_result['A'], result)
