import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn_pandas.base import *
from sklearn_pandas.column_filter import *
from sklearn_pandas.category_transform import *
from sklearn_pandas.numeric_transform import *


def simple_encoding_pipeline(rare_value_threshold=0.05):
    pipeline = Pipeline(steps=[
        ('impute', StringImputer()),
        ('rare_values', BundleRareValues(threshold=rare_value_threshold)),
        ('encode', CategoricalEncoder()),
        ('as_float', TypeCast(dtype=float)),
    ])
    return pipeline


def categorical_aggregate_pipeline(rare_value_threshold=0.05, cat_agg_func='mean', cat_agg_rank=False):
    pipeline = Pipeline(steps=[
        ('impute', StringImputer()),
        ('rare_values', BundleRareValues(threshold=rare_value_threshold)),
        ('encode', CategoricalAggregate(agg_func=cat_agg_func, rank=cat_agg_rank)),
        ('as_float', TypeCast(dtype=float)),
    ])
    return pipeline


def pipeline_eval(pipeline, X, y, base_model=LinearRegression(), eval_func=r2_score):
    eval_pipeline = Pipeline(steps=[
        ('pipeline', pipeline),
        ('model', DataFrameModelTransformer(base_model, output_column_names='y')),
    ])
    yhat = eval_pipeline.fit_transform(X, y)
    return eval_func(yhat, y)


if __name__ == '__main__':
    X = pd.DataFrame({
        'A': ['a', 'a', 'b', 'a', 'a', 'a', 'b', 'a', 'c', 'd', ],
    })

    y = pd.DataFrame({
        'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
    })


    pipeline = simple_encoding_pipeline(rare_value_threshold=0.15)
    pipeline_performance = pipeline_eval(pipeline, X, y)
    print(pipeline_performance)

    pipeline = categorical_aggregate_pipeline(rare_value_threshold=0.05, cat_agg_func='mean', cat_agg_rank=False)
    pipeline_performance = pipeline_eval(pipeline, X, y)
    print(pipeline_performance)

    pipeline = categorical_aggregate_pipeline(rare_value_threshold=0.05, cat_agg_func='min', cat_agg_rank=False)
    pipeline_performance = pipeline_eval(pipeline, X, y)
    print(pipeline_performance)