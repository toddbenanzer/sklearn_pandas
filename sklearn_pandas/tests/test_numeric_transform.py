from unittest import TestCase
from sklearn_pandas.numeric_transform import *


class TestQuantileBinning(TestCase):

    def test_continuous_input(self):
        X = pd.DataFrame({'A': np.linspace(0, 1, 100000)})
        qb = QuantileBinning(nbins=10)
        df_out = qb.fit_transform(X)
        self.assertEqual('(-inf, 0.1]', str(df_out.loc[0,'A']))

        X = pd.DataFrame({'A': np.linspace(0, 1, 100000)})
        qb = QuantileBinning(nbins=5)
        df_out = qb.fit_transform(X)
        self.assertEqual('(-inf, 0.2]', str(df_out.loc[0,'A']))

    def test_repeated_value(self):
        X = pd.DataFrame({'A': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]})
        qb = QuantileBinning(nbins=100)
        df_out = qb.fit_transform(X)
        self.assertEqual('(-inf, 1.0]', str(df_out.iloc[0, 0]))
        self.assertEqual('(1.95, 2.0]', str(df_out.iloc[-1, 0]))


class TestWinsorizeTransform(TestCase):
    def test_continuous_input(self):
        X = pd.DataFrame({'A': np.linspace(0, 1, 100000)})
        wt = WinsorizeTransform(clip_p=0.05)
        df_out = wt.fit_transform(X)
        self.assertEqual('0.05000000000000001', str(df_out.iloc[0, 0]))
        self.assertEqual('0.9499999999999998', str(df_out.iloc[-1, 0]))


class TestPandasRobustScaler(TestCase):
    def test_basic(self):
        X = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        scaler = PandasRobustScaler()
        pd.testing.assert_frame_equal(scaler.fit_transform(X), pd.DataFrame({'A': [-1.0, -0.5, 0.0, 0.5, 1.0]}))


class TestPandasStandardScaler(TestCase):
    def test_basic(self):
        X = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        scaler = PandasStandardScaler()
        pd.testing.assert_frame_equal(scaler.fit_transform(X), pd.DataFrame({'A': [-1.414213562373095, -0.7071067811865475, 0.0, 0.7071067811865475, 1.414213562373095]}))


class TestPandasMinMaxScaler(TestCase):
    def test_basic(self):
        X = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        scaler = PandasMinMaxScaler()
        pd.testing.assert_frame_equal(scaler.fit_transform(X), pd.DataFrame({'A': [0.0, 0.25, 0.5, 0.75, 1.0]}))

    def test_neg_1_to_1(self):
        X = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        scaler = PandasMinMaxScaler(feature_range=(-1, 1))
        pd.testing.assert_frame_equal(scaler.fit_transform(X), pd.DataFrame({'A': [-1.0, -0.5, 0.0, 0.5, 1.0]}))


class TestMissingImputer(TestCase):

    def test_no_missing(self):
        X = pd.DataFrame({'A': [1, 2, 3, ]})
        imputer = MissingImputer()
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1, 2, 3, ]}))

    def test_nan_repalce_mean(self):
        X = pd.DataFrame({'A': [1.0, np.nan, 3.0, ]})
        imputer = MissingImputer(method='mean')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 2.0, 3.0, ]}))

    def test_nan_repalce_zero(self):
        X = pd.DataFrame({'A': [1.0, np.nan, 3.0, ]})
        imputer = MissingImputer(method='zero')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 0.0, 3.0, ]}))

    def test_nan_repalce_median(self):
        X = pd.DataFrame({'A': [1.0, np.nan, 3.0, ]})
        imputer = MissingImputer(method='median')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 2.0, 3.0, ]}))

    def test_inf_repalce_zero(self):
        X = pd.DataFrame({'A': [1.0, np.inf, 3.0, ]})
        imputer = MissingImputer(method='zero')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 0.0, 3.0, ]}))

    def test_inf_repalce_mean(self):
        X = pd.DataFrame({'A': [1.0, np.inf, 3.0, ]})
        imputer = MissingImputer(method='mean')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 2.0, 3.0, ]}))

    def test_ninf_repalce_mean(self):
        X = pd.DataFrame({'A': [1.0, -np.inf, 3.0, ]})
        imputer = MissingImputer(method='mean')
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 2.0, 3.0, ]}))

    def test_ninf_repalce_mean_with_indicators(self):
        X = pd.DataFrame({'A': [1.0, -np.inf, 3.0, ]})
        imputer = MissingImputer(method='mean', create_indicators=True)
        pd.testing.assert_frame_equal(imputer.fit_transform(X), pd.DataFrame({'A': [1.0, 2.0, 3.0, ], 'A_isna': [False, True, False, ]}))

    def test_ninf_repalce_mean_with_indicators_multicolumn(self):
        X = pd.DataFrame({'A': [1.0, -np.inf, 3.0, ], 'B': [np.nan, -np.inf, np.inf, ]})
        imputer = MissingImputer(method='mean', create_indicators=True)
        pd.testing.assert_frame_equal(
            imputer.fit_transform(X),
            pd.DataFrame({'A': [1.0, 2.0, 3.0, ],
                          'A_isna': [False, True, False, ],
                          'B': [0.0, 0.0, 0.0, ],
                          'B_isna': [True, True, True, ]}))
