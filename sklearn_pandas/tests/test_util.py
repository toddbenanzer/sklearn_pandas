from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn_pandas.util import is_dataframe, validate_dataframe


class TestIs_Dataframe(TestCase):

    def test_None(self):
        self.assertFalse(is_dataframe(None))

    def test_dataframe(self):
        df = pd.DataFrame()
        self.assertTrue(is_dataframe(df))

    def test_not_dataframe(self):
        df = [[1, 4, 5], [-5, 8, 9]]
        self.assertFalse(is_dataframe(df))

        df = np.array([[1, 4, 5], [-5, 8, 9]])
        self.assertFalse(is_dataframe(df))


class TestValidate_dataframe(TestCase):

    def test_None(self):
        self.assertRaises(TypeError, validate_dataframe, None)

    def test_dataframe(self):
        df = pd.DataFrame()
        validate_dataframe(df)

    def test_not_dataframe(self):
        df = [[1, 4, 5], [-5, 8, 9]]
        self.assertRaises(TypeError, validate_dataframe, df)

        df = np.array([[1, 4, 5], [-5, 8, 9]])
        self.assertRaises(TypeError, validate_dataframe, df)


