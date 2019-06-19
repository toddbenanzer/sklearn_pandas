from unittest import TestCase
import pandas as pd
from sklearn_pandas.column_filter import *


class TestColumnSelector(TestCase):

    def test_ColumnSelector_all_columns(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        expected_out = pd.DataFrame({'A': [1, 1, ]})
        cs = ColumnSelector(columns=None)
        pd.testing.assert_frame_equal(df, cs.fit_transform(df))

    def test_ColumnSelector_select_columns(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        expected_out = pd.DataFrame({'A': [1, 1, ]})
        cs = ColumnSelector(columns=['A'])
        pd.testing.assert_frame_equal(expected_out, cs.fit_transform(df))

    def test_ColumnSelector_reverse_order(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        expected_out = pd.DataFrame({'B': [1, 1, ], 'A': [1, 1, ]})
        cs = ColumnSelector(columns=['B', 'A'])
        pd.testing.assert_frame_equal(expected_out, cs.fit_transform(df))


class TestDropColumns(TestCase):

    def test_DropColumns_no_columns(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        dc = DropColumns(columns=None)
        pd.testing.assert_frame_equal(df, dc.fit_transform(df))

    def test_DropColumns_one_columns(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        expected_df = pd.DataFrame({'A': [1, 1, ], })
        dc = DropColumns(columns=['B'])
        pd.testing.assert_frame_equal(expected_df, dc.fit_transform(df))

    def test_DropColumns_all_columns(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 1, ]})
        expected_df = pd.DataFrame({'A': [1, 1, ], })
        dc = DropColumns(columns=['A', 'B'])
        pd.testing.assert_frame_equal(expected_df.drop(columns='A'), dc.fit_transform(df))


class TestColumnSearchSelect(TestCase):

    def test_ColumnSearchSelect_all_columns(self):
        df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
        css = ColumnSearchSelect()
        pd.testing.assert_index_equal(df.columns, css.fit_transform(df).columns)

    def test_ColumnSearchSelect_a_prefix(self):
        df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
        expected_df = pd.DataFrame(columns=['aa', 'ab', ])
        css = ColumnSearchSelect(prefix='a')
        pd.testing.assert_index_equal(expected_df.columns, css.fit_transform(df).columns)

    def test_ColumnSearchSelect_a_suffix(self):
        df = pd.DataFrame(columns=['aa', 'ab', 'ba', 'bb', 'cc'])
        expected_df = pd.DataFrame(columns=['aa', 'ba', ])
        css = ColumnSearchSelect(suffix='a')
        pd.testing.assert_index_equal(expected_df.columns, css.fit_transform(df).columns)


class TestUniqueValueFilter(TestCase):

    def test_UniqueValueFilter_keep_all(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
        uvf = UniqueValueFilter(min_unique_values=1)
        pd.testing.assert_frame_equal(df, uvf.fit_transform(df))

    def test_UniqueValueFilter_keep_some(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
        expected_df = pd.DataFrame({'B': [1, 2, ]})
        uvf = UniqueValueFilter(min_unique_values=2)
        pd.testing.assert_frame_equal(expected_df, uvf.fit_transform(df))

    def test_UniqueValueFilter_keep_none(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': [1, 2, ]})
        expected_df = pd.DataFrame({'B': [1, 2, ]}).drop(columns='B')
        uvf = UniqueValueFilter(min_unique_values=3)
        pd.testing.assert_frame_equal(expected_df, uvf.fit_transform(df))


class TestColumnByType(TestCase):

    def test_selector_numerics(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
        expected_df = pd.DataFrame({'A': [1, 1, ]})
        filter = ColumnByType(numerics=True)
        pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))

    def test_selector_strings(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
        expected_df = pd.DataFrame({'B': ['a', 'b', ],})
        filter = ColumnByType(strings=True)
        pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))

    def test_selector_booleans(self):
        df = pd.DataFrame({'A': [1, 1, ], 'B': ['a', 'b', ], 'C': [True, False, ], })
        expected_df = pd.DataFrame({'C': [True, False, ],})
        filter = ColumnByType(booleans=True)
        pd.testing.assert_frame_equal(expected_df, filter.fit_transform(df))


class TestCorrelationFilter(TestCase):

    def test_UniqueValueFilter_base(self):
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

