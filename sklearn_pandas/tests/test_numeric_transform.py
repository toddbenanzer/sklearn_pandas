from unittest import TestCase
from sklearn_pandas.transformers.numeric_transform import *


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


class TestAggByGroupTransform(TestCase):

    def test_mean(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['A', 'A', 'A', 'B', 'B']})
        expected_df = pd.DataFrame({'A_mean_by_B': [2.0, 2.0, 2.0, 4.5, 4.5], })
        transform = AggByGroupTransform(groupby_vars=['B',], metric_vars=['A',], agg_func='mean')
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)

    def test_max(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['A', 'A', 'A', 'B', 'B']})
        expected_df = pd.DataFrame({'A_max_by_B': [3, 3, 3, 5, 5], })
        transform = AggByGroupTransform(groupby_vars=['B',], metric_vars=['A',], agg_func='max')
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)


class TestPandasPCA(TestCase):

    def test_basic(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 3.0, 5.0, 1.0, 7.0, 9.0, 9.0, 10.0,],
            'B': [1.0, 5.0, 3.0, 4.0, 5.0, 2.0, 7.0, 8.0, 4.0, 4.0,],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, 3.0, 7.0, 8.0, 4.0, 7.0,],
            'D': [1.0, 2.0, 3.0, 9.0, 5.0, 4.0, 7.0, 8.0, 5.0, 10.0,],
            'E': [1.0, 7.0, 3.0, 4.0, 5.0, 5.0, 7.0, 1.0, 9.0, 10.0,],
        })
        expected_df = pd.DataFrame({
            'pca_000': [1.6641998841997352, 0.6564859866348204, 0.7702490604695482, 0.10057291223511106,
                        -0.12370176326063885, 0.8743187079588026, -1.017652586990826, -1.2421388904646986,
                        -0.381447967313988, -1.300885343467866],
        })
        transform = PandasPCA(n_components=0.5)
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)

    def test_more_variance(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'B': [1.0, 5.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 3.0, 9.0, 5.0, ],
            'E': [1.0, 7.0, 3.0, 4.0, 5.0, ],
        })
        expected_df = pd.DataFrame({
            'pca_000': [1.5394593228774092, -0.0994291535814626, 0.17852918372763235, -0.4361583976014344, -1.1824009554221444],
            'pca_001': [-0.3386113890536666, 1.7737167618390517, -0.36955057703541505, -0.6650650307328062, -0.40048976501716366],
            'pca_002': [-0.22077285553779147, 0.2098635084745946, -0.6040637672495105, 1.6023277932739368, -0.9873546789612294],
        })
        transform = PandasPCA(n_components=0.95)
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)

    def test_specified_columns(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'B': [1.0, 5.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 3.0, 9.0, 5.0, ],
            'E': [1.0, 7.0, 3.0, 4.0, 5.0, ],
        })
        expected_df = pd.DataFrame({
            'pca_000': [1.5394593228774092, -0.0994291535814626, 0.17852918372763235, -0.4361583976014344, -1.1824009554221444],
            'pca_001': [-0.3386113890536666, 1.7737167618390517, -0.36955057703541505, -0.6650650307328062, -0.40048976501716366],
            'pca_002': [-0.22077285553779147, 0.2098635084745946, -0.6040637672495105, 1.6023277932739368, -0.9873546789612294],
        })
        transform = PandasPCA(n_components=3)
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)


class TestPandasKernelPCA(TestCase):

    def test_basic(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 3.0, 5.0, 1.0, 7.0, 9.0, 9.0, 10.0,],
            'B': [1.0, 5.0, 3.0, 4.0, 5.0, 2.0, 7.0, 8.0, 4.0, 4.0,],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, 3.0, 7.0, 8.0, 4.0, 7.0,],
            'D': [1.0, 2.0, 3.0, 9.0, 5.0, 4.0, 7.0, 8.0, 5.0, 10.0,],
            'E': [1.0, 7.0, 3.0, 4.0, 5.0, 5.0, 7.0, 1.0, 9.0, 10.0,],
        })
        expected_df = pd.DataFrame({
            'pca_000': [-3.1672583539488492, -1.2494056424113635, -1.4659163208430037, -0.1914075343243975,
                        0.2354257122628417, -1.6639787432312583, 1.9367677453686871, 2.364003756364218,
                        0.7259610294066303, 2.4758083513564952],
        })
        transform = PandasKernelPCA(n_components=1)
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)

    def test_more_variance(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'B': [1.0, 5.0, 3.0, 4.0, 5.0, ],
            'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
            'D': [1.0, 2.0, 3.0, 9.0, 5.0, ],
            'E': [1.0, 7.0, 3.0, 4.0, 5.0, ],
        })
        expected_df = pd.DataFrame({
            'pca_000': [3.1438279113038132, -0.20305060587275298, 0.36458581428196135, -0.890706836973131, -2.4146562827398905],
            'pca_001': [-0.3949178565788713, 2.068661729663432, -0.431001810624807, -0.7756562977890109, -0.4670857646707431],
            'pca_002': [0.18724656295264966, -0.17799389583166125, 0.5123313912218616, -1.359000277833924, 0.8374162194910738],
        })
        transform = PandasKernelPCA(n_components=3)
        pd.testing.assert_frame_equal(transform.fit_transform(df), expected_df)

