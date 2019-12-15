from unittest import TestCase
from sklearn_pandas.transformers.feature_selection import *


class TestPandasSelectKBest(TestCase):
    def get_Xy_simple(self):
        X = pd.DataFrame({
            'A': [1.0, 1.0, 1.0, 1.0, 2.0, ],
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 2.0, 2.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        y = [1.0, 2.0, 3.0, 4.0, 5.0, ]
        return X, y

    def test_continuous_input(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectKBest()
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)

    def test_continuous_input_k3(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectKBest(k=3)
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)

    def test_continuous_input_pctk(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectKBest(k=0.5)
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)

    def test_continuous_input_all(self):
        X, y = self.get_Xy_simple()
        expected_df = X
        transform = PandasSelectKBest(k='all')
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)

    def test_continuous_input_pearson(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectKBest(method='pearson')
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)

    def test_continuous_input_spearman(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectKBest(method='spearman')
        out_df = transform.fit_transform(X, y)
        pd.testing.assert_frame_equal(out_df, expected_df)


class TestPandasSelectThreshold(TestCase):
    def get_Xy_simple(self):
        X = pd.DataFrame({
            'A': [1.0, 1.0, 1.0, 1.0, 2.0, ],
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 2.0, 2.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        y = [1.0, 2.0, 3.0, 4.0, 5.0, ]
        return X, y

    def test_continuous_input(self):
        X, y = self.get_Xy_simple()
        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
        })
        transform = PandasSelectThreshold(pct=0.20)
        out_df = transform.fit_transform(X, y)
        print(out_df)
        pd.testing.assert_frame_equal(out_df, expected_df)