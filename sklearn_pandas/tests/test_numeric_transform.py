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
