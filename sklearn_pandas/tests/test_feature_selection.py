import pytest
from sklearn_pandas.transformers.feature_selection import *


def get_Xy_simple():
    X = pd.DataFrame({
        'A': [1.0, 1.0, 1.0, 1.0, 2.0, ],
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
        'D': [1.0, 2.0, 2.0, 2.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    y = [1.0, 2.0, 3.0, 4.0, 5.0, ]
    return X, y


def test_continuous_input_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    transform = PandasSelectKBest()
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def test_continuous_input_k3_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    transform = PandasSelectKBest(k=3)
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def test_continuous_input_pctk_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    transform = PandasSelectKBest(k=0.5)
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def test_continuous_input_all_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = X
    transform = PandasSelectKBest(k='all')
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def test_continuous_input_pearson_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    transform = PandasSelectKBest(method='pearson')
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def test_continuous_input_spearman_PandasSelectKBest():
    X, y = get_Xy_simple()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.0, ],
        'C': [1.0, 2.0, 3.0, 3.0, 5.0, ],
        'E': [-1.0, -1.0, -3.0, -4.0, -5.0, ],
    })
    transform = PandasSelectKBest(method='spearman')
    out_df = transform.fit_transform(X, y)
    pd.testing.assert_frame_equal(out_df, expected_df)


def get_Xy_simple_PandasSelectThreshold():
    X = pd.DataFrame({
        'A': [1.0, 1.001, 1.0, 1.0, 1.001, ],
        'B': [1.0, 2.0, 3.0, 4.0, 5.01, ],
        'C': [1.0, 2.0, 3.0, 3.0, 2.0, ],
        'D': [1.0, 2.0, 2.0, 2.0, 5.0, ],
        'E': [-1.0, -2.0, -3.0, -4.0, -5.01, ],
    })
    y = [1.0, 2.0, 3.0, 4.0, 5.0, ]
    return X, y


def test_continuous_input_PandasSelectThreshold():
    X, y = get_Xy_simple_PandasSelectThreshold()
    expected_df = pd.DataFrame({
        'B': [1.0, 2.0, 3.0, 4.0, 5.01, ],
        'E': [-1.0, -2.0, -3.0, -4.0, -5.01, ],
    })
    transform = PandasSelectThreshold(pct=0.95)
    out_df = transform.fit_transform(X, y)
    #print('********************')
    #print(transform.imp)
    #print('********************')
    #print(out_df)
    pd.testing.assert_frame_equal(out_df, expected_df)
