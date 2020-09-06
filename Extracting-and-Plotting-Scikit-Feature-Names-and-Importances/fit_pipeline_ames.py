from numpy.ma import MaskedArray
import sklearn.utils.fixes

sklearn.utils.fixes.MaskedArray = MaskedArray 

import re
import time
import datetime

import numpy as np  
import pandas as pd  

import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
import category_encoders as ce
from catboost import CatBoostRegressor

import multiprocessing


#### globals ####

DEP_VAR = 'SalePrice'
n_threads = multiprocessing.cpu_count()
MAX_OH_CARDINALITY = 10
N_ESTIMATORS = 1000
SEED = 5


#### load data ####

train_df = pd.read_csv('../kaggle/input/house-prices-advanced-regression-techniques/train.csv')\
            .set_index('Id')\
            .fillna(np.nan)

# split the dependent variable from the features
y_train = train_df.pop(DEP_VAR)
log_y_train = np.log(y_train)


#### create column transformer ####

select_numeric_features = make_column_selector(dtype_include=np.number)
numeric_pipeline = make_pipeline(SimpleImputer(strategy='median', add_indicator=True))  


def select_oh_features(df):

    oh_features =\
        df\
        .select_dtypes(['object', 'category'])\
        .apply(lambda col: col.nunique())\
        .loc[lambda x: x <= MAX_OH_CARDINALITY]\
        .index\
        .tolist()

    return oh_features


oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), 
                            OneHotEncoder(handle_unknown='ignore'))

def select_hc_features(df):

    hc_features =\
        df\
        .select_dtypes(['object', 'category'])\
        .apply(lambda col: col.nunique())\
        .loc[lambda x: x > MAX_OH_CARDINALITY]\
        .index\
        .tolist()

    return hc_features

hc_pipeline = make_pipeline(ce.GLMMEncoder())


column_transformer = ColumnTransformer(transformers=\
                                       [('numeric_pipeline',
                                         numeric_pipeline, 
                                         select_numeric_features),\
                                        ('oh_pipeline', 
                                         oh_pipeline, 
                                         select_oh_features),\
                                        ('hc_pipeline', 
                                         hc_pipeline, 
                                         select_hc_features)
                                       ],\
                                       n_jobs=n_threads,
                                       remainder='drop')

#### create pipeline ####

cat = CatBoostRegressor(thread_count=n_threads, n_estimators=N_ESTIMATORS, 
                        random_state=SEED, verbose=False)


pipe = Pipeline(steps=[('column_transformer', column_transformer),\
                       ('variancethreshold', VarianceThreshold(threshold=0.0)),\
                       ('selectpercentile', SelectPercentile(f_regression, percentile=90)),\
                       ('model', cat)])

_ = pipe.fit(train_df, log_y_train)