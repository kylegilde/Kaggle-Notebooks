import numpy as np  
import pandas as pd  

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import VarianceThreshold

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import multiprocessing


#### globals ####

n_threads = multiprocessing.cpu_count()
N_ESTIMATORS = 1000
np.random.seed(123)


#### create column transformer ####

na_transformer = FunctionTransformer(lambda x: x.fillna(np.nan))

select_numeric_features = make_column_selector(dtype_include=np.number)
numeric_pipe = make_pipeline(na_transformer, 
                             SimpleImputer(strategy='median', add_indicator=True))  

select_oh_features = make_column_selector(dtype_exclude=np.number)
oh_pipe = make_pipeline(na_transformer, 
                        SimpleImputer(strategy='constant'), 
                        OneHotEncoder(handle_unknown='ignore'))

column_transformer = \
    ColumnTransformer([('numeric_pipe', numeric_pipe, select_numeric_features),\
                       ('oh_pipe', oh_pipe, select_oh_features)],
                      n_jobs=n_threads)

#### create model ####

model = CatBoostRegressor(thread_count=n_threads, 
                          n_estimators=N_ESTIMATORS, 
                          verbose=False)

#### create pipeline ####

pipe = Pipeline([('column_transformer', column_transformer),\
                 ('variancethreshold', VarianceThreshold(threshold=0.0)),\
                 ('model', model)])