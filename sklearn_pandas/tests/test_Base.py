from unittest import TestCase
import pandas as pd
from datetime import datetime, date
from sklearn.dummy import DummyRegressor
from sklearn_pandas.transformers.base import *


class TestModelTransformer(TestCase):
    def test_transform(self):
        df = pd.DataFrame({'A': [1, 2, ]})
        expected_out = pd.DataFrame({'B': [0.0, 0.0, ]})
        y = [0.0, 3.0, ]
        model = DummyRegressor(strategy='constant', constant=0.0)
        model_transform = DataFrameModelTransformer(model, output_column_names=['B'])
        model_transform.fit(df, y)
        model_output = model_transform.transform(df)
        pd.testing.assert_frame_equal(expected_out, model_output)


class TestDataFrameFunctionApply(TestCase):
    def test_identity(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        dffa = DataFrameFunctionApply()
        expected_out = df
        pd.testing.assert_frame_equal(expected_out, dffa.fit_transform(df))

    def test_plus_one_prefix(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        dffa = DataFrameFunctionApply(func=lambda x: 2*x, prefix='dub_')
        expected_out = pd.DataFrame({'dub_A': [2, 4, 6], 'dub_B': [8, 10, 12]})
        pd.testing.assert_frame_equal(expected_out, dffa.fit_transform(df))

    def test_plus_one_suffix(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        dffa = DataFrameFunctionApply(func=lambda x: 2*x, suffix='_dub')
        expected_out = pd.DataFrame({'A_dub': [2, 4, 6], 'B_dub': [8, 10, 12]})
        pd.testing.assert_frame_equal(expected_out, dffa.fit_transform(df))


class TestInferType(TestCase):

    def test_passthrough(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        transformer = InferType()
        pd.testing.assert_frame_equal(df, transformer.fit_transform(df))

    def test_datatypes(self):
        df = pd.DataFrame({
            'integer': [1, 2, 3, ],
            'floating': [1.1, 2.2, 3.3, ],
            'string': ['1', '2', '3',],
            'boolean': [True, False, True, ],
            'datetime64': [datetime(2017, 1, 1), datetime(2017, 1, 2), datetime(2017, 1, 3)],
            'date': [date(2017, 1, 1), date(2017, 1, 2), date(2017, 1, 3), ],
        })
        transformer = InferType()
        transformer.fit(df)
        for col in df.columns:
            print(col)
            self.assertEqual(col, transformer.types[col])


