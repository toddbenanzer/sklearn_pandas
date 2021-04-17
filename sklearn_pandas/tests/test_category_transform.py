import pytest
from sklearn_pandas.transformers.category_transform import *


def test_basic_function_StringImputer():
    df = pd.DataFrame({'A': ['', 'hey', None, ], })
    df_expected = pd.DataFrame({'A': ['Blank', 'hey', 'None', ], })
    transform = StringImputer()
    pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


def test_single_value_replace_BundleRareValues():
    df = pd.DataFrame({
        'A': ['a'] * 20 + ['b'] * 1
    })
    df_expected = pd.DataFrame({
        'A': ['a'] * 20 + ['Other'] * 1
    })
    transform = BundleRareValues()
    pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


def test_no_common_value_BundleRareValues():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e', ]
    })
    df_expected = pd.DataFrame({
        'A': ['Other', ] * 5
    })
    transform = BundleRareValues(threshold=0.50)
    pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


def test_dual_value_replace_BundleRareValues():
    df = pd.DataFrame({
        'A': ['a'] * 20 + ['b'] * 1 + ['c'] * 1,
        'B': ['c'] * 20 + ['a'] * 1 + ['c'] * 1,
    })
    df_expected = pd.DataFrame({
        'A': ['a'] * 20 + ['Other'] * 1 + ['Other'] * 1,
        'B': ['c'] * 20 + ['Other'] * 1 + ['c'] * 1,
    })
    transform = BundleRareValues()
    pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


def test_simple_case_CategoricalEncoder():
    df = pd.DataFrame({
        'A': ['a', 'a', 'a', 'b', 'a', ]
    })
    df_expected = pd.DataFrame({
        'A_a': [True, True, True, False, True, ],
        'A_b': [False, False, False, True, False, ],
    })
    transform = CategoricalEncoder()
    pd.testing.assert_frame_equal(transform.fit_transform(df), df_expected)


def test_simple_case_CategoricalAggregate():
    df = pd.DataFrame({
        'A': ['a', 'a', 'a', 'b', 'b', ]
    })

    y = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, ]
    })

    df_expected = pd.DataFrame({
        'A': [2.0, 2.0, 2.0, 4.5, 4.5, ]
    })

    transform = CategoricalAggregate(prefix='', suffix='')
    pd.testing.assert_frame_equal(transform.fit_transform(df, y), df_expected)


def test_y_as_vector_CategoricalAggregate():
    df = pd.DataFrame({
        'A': ['a', 'a', 'a', 'b', 'b', ]
    })

    y = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, ]
    })

    df_expected = pd.DataFrame({
        'A': [2.0, 2.0, 2.0, 4.5, 4.5, ]
    })

    transform = CategoricalAggregate(prefix='', suffix='')
    pd.testing.assert_frame_equal(
        transform.fit_transform(df, y['y']), df_expected)


def test_max_encoding_CategoricalAggregate():
    df = pd.DataFrame({
        'A': ['a', 'a', 'a', 'b', 'b', ]
    })

    y = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, ]
    })

    df_expected = pd.DataFrame({
        'A': [3, 3, 3, 5, 5, ]
    })

    transform = CategoricalAggregate(agg_func='max', prefix='', suffix='')
    pd.testing.assert_frame_equal(transform.fit_transform(df, y), df_expected)
