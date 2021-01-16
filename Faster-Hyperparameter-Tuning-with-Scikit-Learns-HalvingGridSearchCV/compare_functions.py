import numpy as np  
import pandas as pd  

def find_rank(full_results, halving_results):
    
    best_halving_results_df =\
        pd.DataFrame(halving_results.best_params_, index=[0])\
        .rename(columns=lambda x: 'param_' + x)
    
    return pd.DataFrame(full_results.cv_results_)\
                .filter(regex='param_|rank_test_score')\
                .merge(best_halving_results_df)\
                .loc[:, 'rank_test_score']\
                .values


def compare_cv_best_params(full_results, *halving_results):
    
    cv_results = [full_results] + list(halving_results)
    
    df_list = []
    
    for cv_result in cv_results:
        
        best_params_score =\
            pd.DataFrame(cv_result.best_params_, 
                         index=[cv_result.__class__.__name__])\
            .assign(RMSLE=-cv_result.best_score_,
                    full_grid_search_rank=find_rank(full_results, cv_result) if cv_result != full_results else np.nan)\
            .pipe(lambda df: pd.concat([df.iloc[:, -2:], df.iloc[:, :-2]], axis=1))
        
        
        df_list.append(best_params_score)
        
    return pd.concat(df_list).reset_index()  