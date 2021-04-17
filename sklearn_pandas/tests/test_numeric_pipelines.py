import pytest
import pandas as pd
import numpy as np
from sklearn_pandas.pipelines.numeric_pipelines import *
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn_pandas.util import recip1p


def get_standard_X_y():
    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 1, 2, 3, 2, 1, ],
        'B': [np.nan, 1, 1, 1, np.nan, 1, 2, 1, np.nan, np.inf, ],
    })

    y = pd.DataFrame({
        'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, ],
    })

    return X, y


def test_identity_pipeline():
    X, y = get_standard_X_y()
    pipeline = identity_pipeline()
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.5568276244114114


def test_quantile_pipeline():
    X, y = get_standard_X_y()
    pipeline = quantile_pipeline(nbins=2)
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.5423340961098402


def test_winsorize_pipeline():
    X, y = get_standard_X_y()
    pipeline = winsorize_pipeline(clip_p=0.40)
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.27734375


def test_power_pipeline():
    X, y = get_standard_X_y()
    pipeline = power_pipeline(power_list=(1, 2, ))
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.5699781460571843


def test_power_pipeline2():
    X, y = get_standard_X_y()
    pipeline = power_pipeline(power_list=(1, 0.5, ))
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.5604282125997029


def test_function_list_pipeline():
    X, y = get_standard_X_y()
    pipeline = function_list_pipeline()
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.5568276244114114


def test_function_list_pipeline2():
    X, y = get_standard_X_y()
    pipeline = function_list_pipeline(function_list=(
        identity, np.log1p, recip1p, np.sign, np.abs, ))
    pipeline_performance = pipeline_eval(pipeline, X, y)
    assert pipeline_performance == 0.6697943743393496


def test_eval_metric_list():
    eval_func_list = [mean_squared_error, r2_score,
                      explained_variance_score, mean_absolute_error, median_absolute_error]
    eval_result_list = [0.12163589858023631, 0.6697943743393496,
                        0.6697943743393495, 0.2812552288021514, 0.2548840169305322]
    for eval_func, eval_result in zip(eval_func_list, eval_result_list):
        X, y = get_standard_X_y()
        pipeline = function_list_pipeline(function_list=(
            identity, np.log1p, recip1p, np.sign, np.abs, ))
        pipeline_performance = pipeline_eval(
            pipeline, X, y, base_model=LinearRegression(), eval_func=eval_func)
        assert pipeline_performance == eval_result
