import pandas as pd


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logger.info('\nTimings: %r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def is_dataframe(possible_dataframe):
    """Simple helper that returns True if an input is a pandas dataframe."""
    return issubclass(pd.DataFrame, type(possible_dataframe))


def validate_dataframe_input(possible_dataframe):
    """Validate that input is a pandas dataframe and raise an error if it is not. Stays silent if it is."""
    if is_dataframe(possible_dataframe) is False:
        raise TypeError('This transformer requires a pandas dataframe and you passed in a {}'.format(type(possible_dataframe)))


def eval_rows(X, func, dtype = object):
    if hasattr(X, "apply"):
        return X.apply(func, axis = 1)
    nrow = X.shape[0]
    y = np.empty(shape = (nrow, ), dtype = dtype)
    for i in range(0, nrow):
        y[i] = func(X[i])
    return y


def check_df(input_data, ignore_none=False, single_column=False):
    """Convert non dataframe inputs into dataframes (or series).
    Args:
        input_data (:obj:`pd.DataFrame`, :obj:`np.ndarray`, list): input
            to convert
        ignore_none (bool): allow None to pass through check_df
        single_column (bool): check if frame is of a single column and return
            series
    Returns:
        :obj:`DataFrame <pd.DataFrame>`: Converted and validated input \
            dataframes
    Raises:
        ValueError: Invalid input type
        ValueError: Input dataframe must only have one column
    """
    if input_data is None and ignore_none:
        return None

    ret_df = None
    if isinstance(input_data, pd.DataFrame):
        if len(input_data.columns) > len(set(input_data.columns)):
            warnings.warn(
                "Columns are not all uniquely named, automatically resolving"
            )
            input_data.columns = pd.io.parsers.ParserBase(
                {"names": input_data.columns}
            )._maybe_dedup_names(input_data.columns)
        ret_df = input_data
    elif isinstance(input_data, pd.Series):
        ret_df = input_data.to_frame()
    elif isinstance(input_data, np.ndarray) or isinstance(
        input_data, (list, tuple)
    ):
        ret_df = pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Invalid input type, neither pd.DataFrame, pd.Series, np.ndarray, "
            "nor list"
        )

    if single_column and len(ret_df.columns) != 1:
        raise ValueError("Input Dataframe must have only one column")

    return ret_df

