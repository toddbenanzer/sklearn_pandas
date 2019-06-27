import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn_pandas.base import *
from sklearn_pandas.column_filter import *
from sklearn_pandas.category_transform import *
from sklearn_pandas.numeric_transform import *
from sklearn_pandas.util import identity, recip1p


def identity_pipeline():
    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ]),
            Pipeline(steps=[
                ('impute_values', MissingImputer(create_indicators=False, indicator_only=False)),
            ]),
        ])),
        ('as_float', TypeCast(dtype=float)),
        # ('debug', PrintToScreen(max_rows=20, max_cols=20)),
    ])
    return pipeline


def quantile_pipeline(nbins=5):
    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ]),
            Pipeline(steps=[
                ('impute_values', MissingImputer(create_indicators=False, indicator_only=False)),
                ('quantile_binning', QuantileBinning(nbins=nbins, suffix='_binned')),
                ('encode', CategoricalEncoder()),
            ]),
        ])),
        ('as_float', TypeCast(dtype=float)),
        # ('debug', PrintToScreen(max_rows=20, max_cols=20)),
    ])
    return pipeline


def winsorize_pipeline(clip_p=0.05):
    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ]),
            Pipeline(steps=[
                ('impute_values', MissingImputer(create_indicators=False, indicator_only=False)),
                ('winsorize', WinsorizeTransform(clip_p=clip_p, suffix='_winsorize')),
            ]),
        ])),
        ('as_float', TypeCast(dtype=float)),
        # ('debug', PrintToScreen(max_rows=20, max_cols=20)),
    ])
    return pipeline


def power_pipeline(power_list=(1,)):

    def func_factory(p):
        def func(x):
            return np.power(x, p)
        return func

    poly_pipelines = [
        Pipeline(steps=[
            ('power_{0}'.format(power),
             DataFrameFunctionApply(func=func_factory(power), suffix='_pow{0}'.format(power), safe_sign=True)),
        ]) for power in power_list
    ]

    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ]),
            Pipeline(steps=[
                ('impute_values', MissingImputer(create_indicators=False, indicator_only=False)),
                ('asfloat', TypeCast(dtype=float)),
                ('polynomial_terms', DataFrameFeatureUnion(poly_pipelines))
            ]),
        ])),
        ('as_float', TypeCast(dtype=float)),
        # ('debug', PrintToScreen(max_rows=20, max_cols=20)),
    ])
    return pipeline


def function_list_pipeline(function_list=(identity,)):

    func_feature_union_pipelines = [
        Pipeline(steps=[
            ('xform_{0}'.format(func.__name__),
             DataFrameFunctionApply(func=func, suffix='_xform_{0}'.format(func.__name__), safe_sign=True)),
        ]) for func in function_list
    ]

    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ]),
            Pipeline(steps=[
                ('impute_values', MissingImputer(create_indicators=False, indicator_only=False)),
                ('asfloat', TypeCast(dtype=float)),
                ('func_feature_union', DataFrameFeatureUnion(func_feature_union_pipelines))
            ]),
        ])),
        ('as_float', TypeCast(dtype=float)),
        # ('debug', PrintToScreen(max_rows=20, max_cols=20)),
    ])
    return pipeline


def pipeline_eval(pipeline, X, y, base_model=LinearRegression(), eval_func=r2_score):
    eval_pipeline = Pipeline(steps=[
        ('pipeline', pipeline),
        ('model', DataFrameModelTransformer(base_model, output_column_names='y')),
    ])
    yhat = eval_pipeline.fit_transform(X, y)
    return eval_func(yhat, y)
