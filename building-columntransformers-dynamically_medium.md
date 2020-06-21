# Building a Scikit-Learn ColumnTransformer Dynamically

Feature engineering can be time consuming part of the machine learning process, especially if you are dealing with many features and different types of features. Over the course of my projects, I've developed some heuristics that allow me to construct a reasonably effective Scikit-Learn ColumnTransformer quickly and dynamically. 

In my post, I will demonstrate 2 techniques. First, I'll show how to select features with logic instead of listing every single column in the code. Second, I will explain the transformer pipelines that I use as my "defaults" when training a new model. I will demonstrate my technique on the Ames, IA house prices dataset, which you can find on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).


Before proceeding, I should note that my post assumes that you have worked with Scikit-Learn before and are familiar with how [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [Pipeline](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline) & [preprocessing classes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) facilitate reproducible feature engineering processes. If you need a refresher, checkout this Scikit-Learn [example](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data).

Let's start by importing the required packages, classes and functions.



<!-- 
 
Lastly, I will collapse all the code into a function that will rely on my default settings and instantiate the ColumnTransformer. 

These rules of thumb work particularly well for tree-based models, which have fewer feature-engineering requirements.
 -->



```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import category_encoders as ce

import multiprocessing

DEP_VAR = 'SalePrice'

train_df = pd.read_csv('./kaggle/input/house-prices-advanced-regression-techniques/train.csv').set_index('Id')
y_train = train_df[DEP_VAR]
train_df.drop(DEP_VAR, axis=1, inplace=True)

test_df =  pd.read_csv('./kaggle/input/house-prices-advanced-regression-techniques/test.csv').set_index('Id')

```

## The Dataset


The Ames training dataset has a relatively small number of observations and a decent amount of features at 79. 43 of these features are categorical, and 36 are numeric. I recommend reading [this notebook](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) if you are interested in some exploratory data analysis on the dataset.




```python
print(train_df.shape)

feature_types = train_df.dtypes.astype(str).value_counts().to_frame('count').rename_axis('datatype').reset_index()

px.bar(feature_types, x='datatype', y='count', color='datatype').update_layout(showlegend=False)
```

    (1460, 79)
    


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v1.53.0
* Copyright 2012-2020, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>




<div>


            <div id="e215dd98-bd12-4c77-a9a5-ad0431c1e578" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("e215dd98-bd12-4c77-a9a5-ad0431c1e578")) {
                    Plotly.newPlot(
                        'e215dd98-bd12-4c77-a9a5-ad0431c1e578',
                        [{"alignmentgroup": "True", "hovertemplate": "datatype=%{x}<br>count=%{y}<extra></extra>", "legendgroup": "object", "marker": {"color": "#636efa"}, "name": "object", "offsetgroup": "object", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["object"], "xaxis": "x", "y": [43], "yaxis": "y"}, {"alignmentgroup": "True", "hovertemplate": "datatype=%{x}<br>count=%{y}<extra></extra>", "legendgroup": "int64", "marker": {"color": "#EF553B"}, "name": "int64", "offsetgroup": "int64", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["int64"], "xaxis": "x", "y": [33], "yaxis": "y"}, {"alignmentgroup": "True", "hovertemplate": "datatype=%{x}<br>count=%{y}<extra></extra>", "legendgroup": "float64", "marker": {"color": "#00cc96"}, "name": "float64", "offsetgroup": "float64", "orientation": "v", "showlegend": true, "textposition": "auto", "type": "bar", "x": ["float64"], "xaxis": "x", "y": [3], "yaxis": "y"}],
                        {"barmode": "relative", "legend": {"title": {"text": "datatype"}, "tracegroupgap": 0}, "margin": {"t": 60}, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "categoryarray": ["object", "int64", "float64"], "categoryorder": "array", "domain": [0.0, 1.0], "title": {"text": "datatype"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "count"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('e215dd98-bd12-4c77-a9a5-ad0431c1e578');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Types of Features


If you are anything like me, the thought of listing 79 features in the code or a configuration file seems like a tedious and unnecessary task. What if there was a way to logically bucket these features by there characteristics?


The key insight that allows you to dynamically construct a ColumnTransformer is understanding that there are 3 broad types of features in non-textual, non-time series datasets:

1. numerical 
2. categorical with moderate-to-low cardinality
3. categorical with high cardinality

Let's take a look at how to dynamically select each feature type and my default transformer pipeline.


## Numerical Features

The sklearn.compose module does come with a handy class called [make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html), and it provides some limited dynamic functionality to select columns by listing dtypes to include or exclude or by selecting the column names that match a regex pattern. To select numeric features, we will instantiate a function to select columns with the np.number datatype, which will match any integer or float columns. When we call the `select_numeric_features` on the training dataset, we see that it correctly selects the 36 `int64` and `float64` columns.



<!-- (Datetime would be a fourth type, but we won't address that here.) -->


```python
select_numeric_features = make_column_selector(dtype_include=np.number)

numeric_features = select_numeric_features(train_df)

print(f'N numeric_features: {len(numeric_features)} \n')
print(', '.join(numeric_features))
```

    N numeric_features: 36 
    
    MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold
    

For numeric features, my default transformation involves using the [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html). I impute missing values with median and setting the `add_indicator` parameter to `True`. Using the median instead of the imputer's mean default guards against the influence of outliers. Using the `add_indicator` functionality calls the [MissingIndicator class](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator), which creates missing-indicator columns for each feature with missing values. In my experience, these extra columns can be of moderate importance to the model when the data is not missing at random.

A few things to note:

- When I construct transformer pipelines, I prefer to use the [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) function as opposed to the Pipeline class. This function replaces the need to explicitly name each Pipeline step and automatically creates the name by taking the lowercased name of the class, e.g. SimpleImputer is named 'simpleimputer'.

- Scikit-Learn imputers require that the missing values are represented with np.nan -- hence, my use of the fillna method.

- If you are going to use a linear model, you are going to want to insert one of the [preprocessors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) to center and scale before the imputer.

- More sophisticated alternatives to the SimpleImputer include the [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer), which requires centering and scalng, or the experimental [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)








```python
train_df.fillna(np.nan, inplace=True)
test_df.fillna(np.nan, inplace=True)

numeric_pipeline = make_pipeline(SimpleImputer(strategy='median', add_indicator=True))  
```

## Categorical with moderate-to-low cardinality

Next, let's discuss how to select and transform the nominal data into numeric form. 

[One-hot (OH) encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/), where an indicator column is created for each unique value, is the most common method. However, the OH transformation may not be suitable for features with high [cardinality](https://en.wikipedia.org/wiki/Cardinality). OH encoding features with many unique values may create too many columns with very low variance, which may take up too much memory or have a negative impact on the performance of linear models. Hence, we may want to limit the features we select for this encoding to ones below a certain threshold of unique values. For the sake of illustration, I'm going to set my limit at 10 values. In reality, we would probably select the threshold to a higher value depending upon the size of your dataset.

Since the [`make_column_selector` isn't capable of detecting cardinality](https://github.com/scikit-learn/scikit-learn/issues/15873), I've developed my own `select_oh_features` custom function. It consists of a piping of pandas methods that do the following:


- Selects the `object` and `category` dtypes from the pandas `DataFrame`

- Counts the number of unique values for those columns

- Subsets the unique value counts if they are less than or equal to `MAX_OH_CARDINALITY`, using an anonymous `lambda` function within the `loc` method

- Extracts the column names from the index and returns them as a list


When we call the function on the training dataset, we see that it selects 40 of the 43 categorical features.



```python
MAX_OH_CARDINALITY = 10

def select_oh_features(df):
    
    hc_features =\
        df\
        .select_dtypes(['object', 'category'])\
        .apply(lambda col: col.nunique())\
        .loc[lambda x: x <= MAX_OH_CARDINALITY]\
        .index\
        .tolist()
        
    return hc_features

oh_features = select_oh_features(train_df)

print(f'N oh_features: {len(oh_features)} \n')
print(', '.join(oh_features))
```

    N oh_features: 40 
    
    MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition
    

I have two default transformations for categorical features with low-to-moderate cardinality: `SimpleImputer` and [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

In the `SimpleImputer`, using the "constant" strategy sets the missing values to "missing_value." I don't set the `add_indicator` parameter to `True` since this would create two exactly the same columns. In the OH encoder, I like to set the `handle_unknown` parameter to "ignore" instead of using the default "error," so that this transformer won't throw an error if it encounters an unknown value in the test dataset and instead sets all of the OH columns to zero if this situation occurs. Because the Ames test dataset contains categorical values not in the training dataset, our ColumnTransformer will fail on the test dataset without using this setting. If you are planning to use a linear model, you will want to set the `drop` parameter so that the features are not perfectly collinear.


<!-- with a "constant" strategy -->


```python
oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore'))
```

## Categorical with high cardinality

To select the features with high cardinality, I've created a similar function that selects the `object` and `category` features with unique value counts greater than the threshold. It selects three features that meet these criteria.


```python
def select_hc_features(df):
    
    hc_features =\
        df\
        .select_dtypes(['object', 'category'])\
        .apply(lambda col: col.nunique())\
        .loc[lambda x: x > MAX_OH_CARDINALITY]\
        .index\
        .tolist()
        
    return hc_features


hc_features = select_hc_features(train_df)

print(f'N hc_features: {len(hc_features)} \n')
print(', '.join(hc_features))
```

    N hc_features: 3 
    
    Neighborhood, Exterior1st, Exterior2nd
    

To transform our features with high cardinality, I could have gone with a more basic approach and used the Scikit-Learn's native [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) or [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder) preprocessor. However, in many cases, these methods are likely to [perform suboptimally](https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b) in your model unless you are dealing with ordinal data. I prefer to use the [Category Encoder](http://contrib.scikit-learn.org/category_encoders) package, which has more than a dozen ways of intelligently encoding highly cardinal features. [This post](https://towardsdatascience.com/all-about-categorical-variable-encoding-30) provides an overview of several of these methods. Most of these are supervised techniques, which use the dependent variable to transform the nominal values into numerical ones. The [TargetEncoder](http://contrib.scikit-learn.org/category_encoders/targetencoder.html) is probably the easiest method to understand, but I prefer to use the [Generalized Linear Mixed Model Encoder](http://contrib.scikit-learn.org/category_encoders/glmm.html), which has "solid statistical theory behind [it]" and "no hyperparameters to tune." Without diving into the [details of GLMMs](https://stats.idre.ucla.edu/other/mult-pkg/introduction-to-generalized-linear-mixed-models/), at its core, this method encodes the nominal values as the coefficents from a one-hot-encoded linear model. The Category Encoder methods handle missing and unknown values by setting them to zero or the mean of the dependent variable. (If these features in the Ames training dataset had any missing values, we would also want to create missing indicators.)




```python
hc_pipeline = make_pipeline(ce.GLMMEncoder())
```

# Putting It All Together

Finally, let's put all the pieces together and instantiate our ColumnTransformer:

- The `transformer` parameter accepts a list of 3-element tuples. Each tuple contains the name of the transformer/pipeline, the instantiated pipelines and the selector functions that we created. 

- If you are dealing with a significant number of features and mulit-thread capability, I would definitely set the `n_jobs` parameter, so that the pipelines can be run in parallel.

- Lastly, I want to call attention to the `remainder` parameter. By default, ColumnTransformer drops any columns not included in `transformers` list. Alternatively, if you have features that require no transformations, you could set this argument to "passthrough," which will not drop these features.


```python
column_transformer = ColumnTransformer(transformers=\
                                       [('numeric_pipeline', numeric_pipeline, select_numeric_features),\
                                        ('oh_pipeline', oh_pipeline, select_oh_features),\
                                        ('hc_pipeline', hc_pipeline, select_hc_features)],
                                       n_jobs = multiprocessing.cpu_count(),
                                       remainder='drop')
```

# Results

As you can see, the OH encodings increased the number of columns from 79 to 254. If we hadn't used the `GLMMEncoder`, we would be dealing with over 300 columns.


```python
X_train = column_transformer.fit_transform(train_df, y_train)
X_test = column_transformer.transform(test_df)

print(X_train.shape)
print(X_test.shape)
```

    (1460, 254)
    (1459, 254)
    

Let's see how are engineered features perform on an GBM regressor.


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error

model = GradientBoostingRegressor(learning_rate=0.025, n_estimators=1000)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
```

With an R-squared value at nearly 0.98, our features explain nearly all of the variation in the training set's dependent variable.
The root mean-squared log error was nearly 0.07.


```python
print(f'R-squared: {r2_score(y_train, y_train_pred)}')
print(f'RMSLE: {np.sqrt(mean_squared_log_error(y_train, y_train_pred))}')
```

    R-squared: 0.9832244795296725
    RMSLE: 0.06623887654531639
    

However, the RMSLE on the test dataset was 0.13249, which means that the model is overfitting. 


```python
submission = pd.DataFrame(dict(Id=test_df.index, 
                               SalePrice=model.predict(X_test)))
submission.to_csv("submission.csv", index=False)
```

Let me know if you found this post helpful or have any ideas for improvement. Stay tuned for further posts on training models with Scikit-Learn ColumnTransformers and Pipelines. Thanks!