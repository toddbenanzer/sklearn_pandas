import pytest
import numpy as np
import pandas as pd
from sklearn_pandas.util import is_pandas_dataframe, validate_dataframe, validate_columns_exist, retain_sign


def test_None_Is_Dataframe():
    assert is_pandas_dataframe(None) is False


def test_dataframe_Is_Dataframe():
    df = pd.DataFrame()
    assert is_pandas_dataframe(df)


def test_not_dataframe_Is_Dataframe():
    df = [[1, 4, 5], [-5, 8, 9]]
    assert is_pandas_dataframe(df) is False

    df = np.array([[1, 4, 5], [-5, 8, 9]])
    assert is_pandas_dataframe(df) is False


def test_None_Validate_dataframe():
    with pytest.raises(TypeError):
        validate_dataframe(None)


def test_dataframe_Validate_dataframe():
    df = pd.DataFrame()
    validate_dataframe(df)
    assert True


def test_not_dataframe_Validate_dataframe():
    df = [[1, 4, 5], [-5, 8, 9]]
    with pytest.raises(TypeError):
        validate_dataframe(df)

    df = ([1, 4, 5], [-5, 8, 9])
    with pytest.raises(TypeError):
        validate_dataframe(df)


def test_None_validate_columns_exist():
    df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
    with pytest.raises(TypeError):
        validate_columns_exist(df)


def test_column_doesnt_exist_validate_columns_exist():
    df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
    with pytest.raises(KeyError):
        validate_columns_exist(df, ['C'])


def test_column_does_exist_validate_columns_exist():
    df = pd.DataFrame({'A': [1, 2, ], 'B': [3, 4, ]})
    validate_columns_exist(df, ['A'])
    assert True


def test_single_value_Retain_sign():
    assert retain_sign(np.sqrt)(-4) == -2.0


def test_array_apply_Retain_sign():
    df = pd.DataFrame({'A': [-1, -4, -9, ]})
    result = df['A'].apply(retain_sign(np.sqrt))
    expected_result = pd.DataFrame({'A': [-1.0, -2.0, -3.0, ], })
    pd.testing.assert_series_equal(expected_result['A'], result)
