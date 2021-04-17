import pytest
from sklearn_pandas.transformers.column_filter import *


def test_ColumnSelector_all_columns_ColumnSelector():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    expected_out = pd.DataFrame({'A': [1, 1, ]})
    cs = ColumnSelector(columns=None)
    pd.testing.assert_frame_equal(df, cs.fit_transform(df))


def test_ColumnSelector_select_columns_ColumnSelector():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    expected_out = pd.DataFrame({'A': [1, 1, ]})
    cs = ColumnSelector(columns=['A'])
    pd.testing.assert_frame_equal(expected_out, cs.fit_transform(df))


def test_ColumnSelector_reverse_order_ColumnSelector():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    expected_out = pd.DataFrame({'B': [1, 1, ], 'A': [1, 1, ]})
    cs = ColumnSelector(columns=['B', 'A'])
    pd.testing.assert_frame_equal(expected_out, cs.fit_transform(df))


def test_DropColumns_no_columns_DropColumns():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    dc = DropColumns(columns=None)
    pd.testing.assert_frame_equal(df, dc.fit_transform(df))


def test_DropColumns_one_columns_DropColumns():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    expected_df = pd.DataFrame({'A': [1, 1, ], })
    dc = DropColumns(columns=['B'])
    pd.testing.assert_frame_equal(expected_df, dc.fit_transform(df))


def test_DropColumns_all_columns_DropColumns():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
    expected_df = pd.DataFrame({'A': [1, 1, ], })
    dc = DropColumns(columns=['A', 'B'])
    pd.testing.assert_frame_equal(
        expected_df.drop(columns='A'), dc.fit_transform(df))


def test_ColumnSearchSelect_all_columns_ColumnSearchSelect():
    df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
    css = ColumnSearchSelect()
    pd.testing.assert_index_equal(df.columns, css.fit_transform(df).columns)


def test_ColumnSearchSelect_a_prefix_ColumnSearchSelect():
    df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
    expected_df = pd.DataFrame(columns=['aa', 'ab', ])
    css = ColumnSearchSelect(prefix='a')
    pd.testing.assert_index_equal(
        expected_df.columns, css.fit_transform(df).columns)


def test_ColumnSearchSelect_a_suffix_ColumnSearchSelect():
    df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
    expected_df = pd.DataFrame(columns=['aa', 'ba', ])
    css = ColumnSearchSelect(suffix='a')
    pd.testing.assert_index_equal(
        expected_df.columns, css.fit_transform(df).columns)


def test_UniqueValueFilter_keep_all_UniqueValueFilter():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
    uvf = UniqueValueFilter(min_unique_values=1)
    pd.testing.assert_frame_equal(df, uvf.fit_transform(df))


def test_UniqueValueFilter_keep_some_UniqueValueFilter():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
    expected_df = pd.DataFrame({'B': [1, 2, ]})
    uvf = UniqueValueFilter(min_unique_values=2)
    pd.testing.assert_frame_equal(expected_df, uvf.fit_transform(df))


def test_UniqueValueFilter_keep_none_UniqueValueFilter():
    df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
    expected_df = pd.DataFrame({'B': [1, 2, ]}).drop(columns='B')
    uvf = UniqueValueFilter(min_unique_values=3)
    pd.testing.assert_frame_equal(expected_df, uvf.fit_transform(df))


def test_selector_numerics_ColumnByType():
    df = pd.DataFrame(
        {'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
    expected_df = pd.DataFrame({'A': [1, 1, ]})
    filter = ColumnByType(numerics=True)
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


def test_selector_strings_ColumnByType():
    df = pd.DataFrame(
        {'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
    expected_df = pd.DataFrame({'B': ['a', 'b', ], })
    filter = ColumnByType(strings=True)
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


def test_selector_booleans_ColumnByType():
    df = pd.DataFrame(
        {'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
    expected_df = pd.DataFrame({'C': [True, False, ], })
    filter = ColumnByType(booleans=True)
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


def test_UniqueValueFilter_base_CorrelationFilter():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, ],
        'B': [1, 2, 3, 4, 4, ],
        'C': [1, 2, 3, 4, 4, ],
        'D': [1, 2, 3, 3, 3, ],
        'E': [5, 2, 1, 4, 3, ],
    })
    expected_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, ],
        'D': [1, 2, 3, 3, 3, ],
        'E': [5, 2, 1, 4, 3, ],
    })
    filter = CorrelationFilter()
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


def test_UniqueValueFilter_spearman_CorrelationFilter():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, ],
        'B': [1, 2, 3, 4, 4, ],
        'C': [1, 2, 3, 4, 4, ],
        'D': [1, 2, 3, 3, 3, ],
        'E': [5, 2, 1, 4, 3, ],
    })
    expected_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, ],
        'D': [1, 2, 3, 3, 3, ],
        'E': [5, 2, 1, 4, 3, ],
    })
    filter = CorrelationFilter(method='spearman')
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


def test_basic_function_PandasSelectKBest():
    X = pd.DataFrame({
        'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, ],
        'C': [1, 2, 3, 4, 2, 3, 4, 3, 2, 1, ],
    })
    y = pd.DataFrame({
        'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
    })
    expected_df = pd.DataFrame({
        'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, ],
        'C': [1, 2, 3, 4, 2, 3, 4, 3, 2, 1, ],
    })

    filter = PandasSelectKBest(k=2)
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(X, y))


def test_basic_function_one_var_PandasSelectKBest():
    X = pd.DataFrame({
        'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, ],
        'C': [1, 2, 3, 4, 2, 3, 4, 3, 2, 1, ],
    })
    y = pd.DataFrame({
        'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
    })
    expected_df = pd.DataFrame({
        'C': [1, 2, 3, 4, 2, 3, 4, 3, 2, 1, ],
    })

    filter = PandasSelectKBest(k=1)
    pd.testing.assert_frame_equal(expected_df, filter.fit_transform(X, y))
