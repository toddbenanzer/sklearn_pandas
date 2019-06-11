# audit table
#     missing
#         count % missing
#         count number missing
#         
#     inferred type
#         does this match training data
#         
#     distribution
#         mix/max bounds violated
#         mean shift
#         variance shift
#         outlier test
#         
# select columns
#     by name
#     by prefix
#     by suffix
#     by regex
# 
# impute missing values
#     
# single column transform
#     apply function
#     aggregate index
#     
# 
# multi column transform


import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from scipy.special import inv_boxcox1p
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn import preprocessing
import itertools
from collections import defaultdict
from datetime import datetime
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
import pandas as pd
import numpy as np
import os
import warnings
import logging
import time
import random
from sklearn.preprocessing import StandardScaler, Imputer
import numpy as np
import pandas as pd
from functools import reduce
import logging
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer
from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin, clone
import pandas as pd
import numpy as np
import os
import warnings
import logging
import time
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, LabelEncoder, LabelBinarizer
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
import logging
import operator
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
logger = logging.getLogger(__name__)

