from unittest import TestCase
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn_pandas.predictor_eval import *
from sklearn_pandas.predictor_eval import _numeric_model_pipeline, _model_score_function_factory_numeric, \
    _model_score_function_factory_categorical, _model_score_function_factory_composite


class Test_numeric_model_pipeline(TestCase):
    def assertHasAttr(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(testBool, msg='obj lacking an attribute. obj: %s, intendedAttr: %s' % (obj, intendedAttr))

    def test_linear_model_build(self):
        pipeline = _numeric_model_pipeline()
        union_step = pipeline.named_steps['numeric_prep'].named_steps['union']
        self.assertHasAttr(union_step.list_of_transformers[0].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'asfloat')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'polynomial_terms')
        polynomial_terms = union_step.list_of_transformers[1].named_steps['polynomial_terms']
        self.assertEqual(len(polynomial_terms.list_of_transformers), 1)

    def test_high_order_model_build(self):
        pipeline = _numeric_model_pipeline(order=20)
        union_step = pipeline.named_steps['numeric_prep'].named_steps['union']
        self.assertHasAttr(union_step.list_of_transformers[0].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'asfloat')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'polynomial_terms')
        polynomial_terms = union_step.list_of_transformers[1].named_steps['polynomial_terms']
        self.assertEqual(len(polynomial_terms.list_of_transformers), 20)

    def test_zero_order_model_build(self):
        pipeline = _numeric_model_pipeline(order=0)
        union_step = pipeline.named_steps['numeric_prep'].named_steps['union']
        self.assertHasAttr(union_step.list_of_transformers[0].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'impute_indicator')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'asfloat')
        self.assertHasAttr(union_step.list_of_transformers[1].named_steps, 'polynomial_terms')
        polynomial_terms = union_step.list_of_transformers[1].named_steps['polynomial_terms']
        self.assertEqual(len(polynomial_terms.list_of_transformers), 0)


class Test_model_score_function_factory_numeric(TestCase):

    def test_simple_scenario(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        })
        expected_series = pd.Series([0.0, 1.0, 0.0], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric()
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_third_order(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        })
        expected_series = pd.Series([0.9175184862983906, 1.0, 0.0], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric(order=3)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_higher_order_response_first_order_fit(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.0006184291898577721, 0.04081632653061229, 0.020408163265306034], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric(order=1)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_higher_order_response_second_order_fit(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.07262844241654909, 0.3480263578785747, 0.020408163265306256], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric(order=3)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_higher_order_response_second_order_fit_mse(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.45441206321589095, 0.3194670846394984, 0.48], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric(order=3, eval_func=mean_squared_error)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)


class Test_model_score_function_factory_categorical(TestCase):

    def test_simple_scenario(self):
        X = pd.DataFrame({
            'A': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', ],
            'B': ['d', None, 'x', None, 'e', 'e', None, 'e', 'e', 'e', ],
            'C': ['f', 'g', 'f', 'g', 'f', 'g', 'f', 'f', 'f', 'f', ],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        })
        expected_series = pd.Series([0.6166666666666667, 0.1266666666666667, 0.09523809523809523], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_categorical()
        pd.testing.assert_series_equal(score_func(X, y), expected_series)


class Test_composite_model_pipeline(TestCase):

    def test_string_scenario(self):
        X = pd.DataFrame({
            'A': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', ],
            'B': ['d', None, 'x', None, 'e', 'e', None, 'e', 'e', 'e', ],
            'C': ['f', 'g', 'f', 'g', 'f', 'g', 'f', 'f', 'f', 'f', ],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        })
        expected_series = pd.Series([0.6166666666666667, 0.1266666666666667, 0.09523809523809523], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_composite()
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_numeric_scenario(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.07262844241654909, 0.3480263578785747, 0.020408163265306256], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_numeric(order=3)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_mixed_scenario(self):
        X = pd.DataFrame({
            'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
            'B': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1,],
            'C': ['a', 'b', 'c', 'b', 'a', 'b', 'b', 'b', 'a', 'b', ],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.07262844241654909, 0.3480263578785747, 0.8299319727891157], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_composite(order=3)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)

    def test_string_scenario2(self):
        X = pd.DataFrame({
            'A': ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', ],
            'B': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', ],
            'C': ['a', 'b', 'c', 'b', 'a', 'b', 'b', 'b', 'a', 'b', ],
        })
        y = pd.DataFrame({
            'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        })
        expected_series = pd.Series([0.0, 0.00874635568513138, 0.8299319727891157], index=['A', 'B', 'C'])
        score_func = _model_score_function_factory_composite(order=3)
        pd.testing.assert_series_equal(score_func(X, y), expected_series)
