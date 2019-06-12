from unittest import TestCase
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn_pandas.base import DataFrameModelTransformer


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