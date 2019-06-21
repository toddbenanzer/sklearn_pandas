import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn_pandas.base import TypeCast, DataFrameFeatureUnion, DataFrameFunctionApply, DataFrameModelTransformer, PrintToScreen, CreateDummyColumn
from sklearn_pandas.numeric_transform import MissingImputer
from sklearn_pandas.category_transform import StringImputer, BundleRareValues, CategoricalEncoder
from sklearn_pandas.column_filter import ColumnByType

def _numeric_prep_pipeline(order):
    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
                ('asfloat', TypeCast(dtype=float)),
            ]),
            Pipeline(steps=[
                ('impute_indicator', MissingImputer(create_indicators=False)),
                ('asfloat', TypeCast(dtype=float)),
                ('polynomial_terms', DataFrameFeatureUnion(
                    [DataFrameFunctionApply(func=lambda x: x) for power in range(min(order, 1))] +
                    [DataFrameFunctionApply(func=lambda x: x ** power,
                                            suffix='_pow{0}'.format(power)) for
                     power in range(2, order + 1)]
                )),
            ]),
        ])),
    ])
    return pipeline


def _string_prep_pipeline():
    pipeline = Pipeline(steps=[
        ('impute', StringImputer()),
        ('rare_values', BundleRareValues(threshold=0.15)),
        ('encode', CategoricalEncoder()),
        ('asfloat', TypeCast(dtype=float)),
    ])
    return pipeline


def _numeric_model_pipeline(base_model=LinearRegression(), order=1):
    pipeline = Pipeline(steps=[
        ('numeric_prep', _numeric_prep_pipeline(order)),
        ('add_dummy', CreateDummyColumn()),
        ('model', DataFrameModelTransformer(base_model, output_column_names='y')),
    ], memory=None)

    return pipeline


def _categorical_model_pipeline(base_model=LinearRegression()):
    pipeline = Pipeline(steps=[
        ('string_prep', _string_prep_pipeline()),
        ('add_dummy', CreateDummyColumn()),
        ('model', DataFrameModelTransformer(base_model, output_column_names='y')),
    ], memory=None)

    return pipeline


def _composite_model_pipeline(base_model=LinearRegression(), order=1):
    pipeline = Pipeline(steps=[
        ('union', DataFrameFeatureUnion([
            Pipeline(steps=[
                ('get_numerics', ColumnByType(numerics=True)),
                ('numeric_prep', _numeric_prep_pipeline(order)),
            ]),
            Pipeline(steps=[
                ('get_strings', ColumnByType(strings=True)),
                ('string_prep', _string_prep_pipeline()),
            ]),
        ])),
        ('add_dummy', CreateDummyColumn()),
        ('model', DataFrameModelTransformer(base_model, output_column_names='y')),
    ], memory=None)
    return pipeline


def _model_score_function_factory_numeric(base_model=LinearRegression(), order=1, eval_func=r2_score):
    def _score_func(X, y):
        pipeline = _numeric_model_pipeline(base_model=base_model, order=order)
        return X.apply(lambda x: eval_func(y, pipeline.fit_transform(x, y)))
    return _score_func


def _model_score_function_factory_categorical(base_model=LinearRegression(), eval_func=r2_score):
    def _score_func(X, y):
        pipeline = _categorical_model_pipeline(base_model=base_model)
        return X.apply(lambda x: eval_func(y, pipeline.fit_transform(x, y)))
    return _score_func


def _model_score_function_factory_composite(base_model=LinearRegression(), order=1, eval_func=r2_score):
    def _score_func(X, y):
        pipeline = _composite_model_pipeline(base_model=base_model, order=order)
        return X.apply(lambda x: eval_func(y, pipeline.fit_transform(x, y)))
    return _score_func
