from unittest import TestCase
import pandas as pd
import numpy as np
from sklearn_pandas.pipelines.numeric_pipelines import *





class TestPipelines(TestCase):

    def get_standard_X_y(self):
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 1, 2, 3, 2, 1, ],
            'B': [np.nan, 1, 1, 1, np.nan, 1, 2, 1, np.nan, np.inf, ],
        })

        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, ],
        })

        return X, y

    def test_identity_pipeline(self):
        X, y = self.get_standard_X_y()
        pipeline = identity_pipeline()
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_quantile_pipeline(self):
        X, y = self.get_standard_X_y()
        pipeline = quantile_pipeline(nbins=2)
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_winsorize_pipeline(self):
        X, y = self.get_standard_X_y()
        pipeline = winsorize_pipeline(clip_p=0.40)
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_power_pipeline(self):
        X, y = self.get_standard_X_y()
        pipeline = power_pipeline(power_list=(1, 2, ))
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_power_pipeline2(self):
        X, y = self.get_standard_X_y()
        pipeline = power_pipeline(power_list=(1, 0.5, ))
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_function_list_pipeline(self):
        X, y = self.get_standard_X_y()
        pipeline = function_list_pipeline()
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)

    def test_function_list_pipeline2(self):
        X, y = self.get_standard_X_y()
        pipeline = function_list_pipeline(function_list=(identity, np.log1p, recip1p, np.sign, np.abs, ))
        pipeline_performance = pipeline_eval(pipeline, X, y)
        print(pipeline_performance)


