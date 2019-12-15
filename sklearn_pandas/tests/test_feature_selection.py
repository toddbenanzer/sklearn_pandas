from unittest import TestCase
from sklearn_pandas.transformers.feature_selection import *


class TestPandasSelectKBest(TestCase):

    def test_continuous_input(self):
        X = pd.DataFrame({
            'A': [1.0, 1.0, 1.0, 1.0, 2.0, ],
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 2.0, 2.0, 5.0, ],
        })
        y = [1.0, 2.0, 3.0, 4.0, 5.0, ]

        expected_df = pd.DataFrame({
            'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
        })
        transform = PandasSelectKBest(k=3)
        pd.testing.assert_frame_equal(transform.fit_transform(X, y), expected_df)
