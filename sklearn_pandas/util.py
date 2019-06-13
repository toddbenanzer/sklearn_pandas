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


def is_dataframe(possible_dataframe):
    """Returns True if object is dataframe."""
    return issubclass(pd.DataFrame, type(possible_dataframe))


def validate_dataframe(possible_dataframe):
    """Validate that input is a pandas dataframe and raise an error if it is not."""
    if not is_dataframe(possible_dataframe):
        raise TypeError('This transformer requires a pandas dataframe and you passed in a {0}'.format(
            type(possible_dataframe)))


def validate_columns_exist(df, columns):
    missing_columns = list(set(columns) - set(df.columns))
    if len(missing_columns) > 0:
        raise KeyError("The DataFrame does not include the columns: %s" % missing_columns)


def retain_sign(func):
    def safe_func(x):
        return np.sign(x) * func(np.abs(x))
    return safe_func