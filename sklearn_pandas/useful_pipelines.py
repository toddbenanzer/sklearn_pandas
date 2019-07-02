from sklearn.pipeline import Pipeline
from sklearn_pandas.transformers.base import *
from sklearn_pandas.transformers.column_filter import *
from sklearn_pandas.transformers.category_transform import *
from sklearn_pandas.transformers.numeric_transform import *

# TODO Create Pipelines for preparation, manufacturing, and variable selection
# Data Prep
# 	Set Correct Type
# 		identify columns that are actually IDs (integer or string columns that are mostly unique: no repeated values
# 	Split by Type
# 		get strings only
# 		get numerics only
# 		get categorical only
# 	Impute Missing Values
# 		create indicators for numeric missing
# 		create indicators for string missing
# 	Create Derived Fields
# 		pass in a list of function transformers
# Data Eval
# 	Calculate variable importance
# Model Performance
# 	Calibrate model


X = pd.DataFrame({
    'A': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', ],
    'B': ['d', None, 'x', None, 'e', 'e', None, 'e', 'e', 'e', ],
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, ],
    'D': [1, 3, 5, 7, 2, 4, 6, 8, 2, 4, ],
    'E': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, ],
})

y = pd.DataFrame({
        'y': [0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
})


fix_column_order_pipeline = Pipeline(steps=[
    ('fix_column_order', DataFrameFixColumnOrder()),
])

drop_trivial_column_pipeline = Pipeline(steps=[
    ('drop_trivial_columns', UniqueValueFilter(min_unique_values=2)),
])

integer_to_string_pipeline = Pipeline(steps=[
    ('integer_to_string', IntegerToString(min_unique_values=5)),
])

get_string_pipeline = Pipeline(steps=[
    ('get_strings', ColumnByType(strings=True)),
])

string_prep_pipeline = Pipeline(steps=[
    ('impute', StringImputer()),
    ('rare_values', BundleRareValues(threshold=0.05)),
    ('encode', CategoricalEncoder()),
    ('calculated_features', DataFrameFeatureUnion([
        Pipeline(steps=[
            ('as_float', TypeCast(dtype=float)),
        ]),
        Pipeline(steps=[
            ('as_float', TypeCast(dtype=float)),
            ('scale', PandasStandardScaler()),
            ('pca', PandasPCA(n_components=0.9, prefix='string_pca_')),
        ]),
    ])),
])

string_pipeline = Pipeline(steps=[
    ('get_strings', get_string_pipeline),
    ('string_prep', string_prep_pipeline),
])

get_numerics_pipeline = Pipeline(steps=[
    ('get_numerics', ColumnByType(numerics=True)),
])

numeric_prep_pipeline = Pipeline(steps=[
    ('union', DataFrameFeatureUnion([
        Pipeline(steps=[
            ('impute_indicator', MissingImputer(create_indicators=True, indicator_only=True)),
            ('as_float', TypeCast(dtype=float)),
        ]),
        Pipeline(steps=[
            ('impute_indicator', MissingImputer(create_indicators=False)),
            ('as_float', TypeCast(dtype=float)),
            ('advanced_features', DataFrameFeatureUnion([
                Pipeline(steps=[
                    ('scale', PandasStandardScaler()),
                    ('pca', PandasPCA(n_components=0.9, prefix='numeric_pca_')),
                ]),
                Pipeline(steps=[
                    ('winsorize', WinsorizeTransform(clip_p=0.95, prefix='winsorize_')),
                ]),
                Pipeline(steps=[
                    ('quantile', QuantileBinning(nbins=10, prefix='quantile_')),
                    ('as_string', TypeCast(dtype=str)),
                    ('rare_values', BundleRareValues(threshold=0.05)),
                    ('encode', CategoricalEncoder()),
                    ('as_float', TypeCast(dtype=float)),
                ]),
            ])),
        ]),
    ])),
])

numeric_pipeline = Pipeline(steps=[
    ('get_numerics', get_numerics_pipeline),
    ('numeric_prep', numeric_prep_pipeline),
])

prepare_data_pipeline = Pipeline(steps=[
    ('fix_column_order', fix_column_order_pipeline),
    ('drop_trivial_columns', drop_trivial_column_pipeline),
    ('integer_to_string', integer_to_string_pipeline),
    ('prep_by_type', DataFrameFeatureUnion([
        string_pipeline,
        numeric_pipeline,
    ])),
])

Xout = prepare_data_pipeline.fit_transform(X, y)
print(Xout.to_string())
