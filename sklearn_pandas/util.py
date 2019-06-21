import time
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info('Eval Time: {0}  {1:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result

    return timed


def ndarry_num_cols(x):
    shape = x.shape
    if len(shape) == 1:
        return 1
    return shape[1]


def array_to_dataframe(x):
    col_names = ['col{0}'.format(i) for i in range(ndarry_num_cols(x))]
    return pd.DataFrame(x, columns=col_names)


def is_pandas_dataframe(obj):
    return issubclass(pd.DataFrame, type(obj))


def is_pandas_series(obj):
    return issubclass(pd.Series, type(obj))


def is_numpy_array(obj):
    return issubclass(np.ndarray, type(obj))


def validate_dataframe(df):
    if is_pandas_dataframe(df):
        return df
    elif is_pandas_series(df):
        return df.to_frame()
    elif is_numpy_array(df):
        return array_to_dataframe(df)
    else:
        raise TypeError('Transformer requires a pandas dataframe not {0}'.format(type(df)))


def validate_columns_exist(df, columns):
    missing_columns = list(set(columns) - set(df.columns))
    if len(missing_columns) > 0:
        raise KeyError("The DataFrame does not include the columns: %s" % missing_columns)


def retain_sign(func):
    def safe_func(x):
        return np.sign(x) * func(np.abs(x))

    return safe_func
