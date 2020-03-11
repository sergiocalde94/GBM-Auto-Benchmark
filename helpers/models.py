import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from typing import Union
from sklearn.metrics import (roc_auc_score,
                             average_precision_score,
                             accuracy_score)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from helpers.data_structures import (Prediction, Metrics)


def convert_to_category_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM needs the categorical data to be `category` dtype
    instead of `object` and `streamlit` isn´t compatible with it yet"""
    df_copy = df.copy()

    df_copy[df_copy.select_dtypes(['object']).columns] = (
        df_copy
        .select_dtypes(['object'])
        .apply(lambda x: x.astype('category'))
    )

    return df_copy


@st.cache(hash_funcs={XGBClassifier: lambda model: 'xgb',
                      CatBoostClassifier: lambda model: 'catboost',
                      LGBMClassifier: lambda model: 'lightgbm'})
def fit_model(model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier],
              features: pd.DataFrame,
              label: pd.DataFrame,
              is_lightgbm: bool = False) -> Union[XGBClassifier,
                                                  CatBoostClassifier,
                                                  LGBMClassifier]:
    """Alias for `fit` methods of Pipeline object and implements
    streamlit caching so we haven't to fit the models all of the
    times that we rerun
    """
    features_copy = (convert_to_category_dtype(features)
                     if is_lightgbm else features.copy())

    with st.spinner('Training XGBoost, LightGBM and CatBoost models...'):
        model.fit(features_copy, label)
    return model


@st.cache(hash_funcs={XGBClassifier: lambda model: 'xgb',
                      CatBoostClassifier: lambda model: 'catboost',
                      LGBMClassifier: lambda model: 'lightgbm'})
def get_predictions(model: Union[XGBClassifier,
                                 CatBoostClassifier,
                                 LGBMClassifier],
                    features_train: pd.DataFrame,
                    features_test: pd.DataFrame,
                    is_lightgbm: bool = False) -> Prediction:
    """Get predictions from `XGBClassifier`, `CatBoostClassifier` or
    `LGBMClassifier` object and returns a namedtuple `Prediction`
    with all of the necessary calculations for get the metrics
    """
    features_train_copy = (convert_to_category_dtype(features_train)
                           if is_lightgbm else features_train.copy())

    features_test_copy = (convert_to_category_dtype(features_test)
                          if is_lightgbm else features_test.copy())

    return Prediction(
        y_score_train=model.predict_proba(features_train_copy)[:, 1],
        y_score=model.predict_proba(features_test_copy)[:, 1],
        y_pred_train=model.predict(features_train_copy),
        y_pred=model.predict(features_test_copy)
    )


@st.cache
def compute_metrics(predictions: Prediction,
                    y_train: pd.Series,
                    y_test: pd.Series) -> Metrics:
    """Compute metrics for the specific model `predictions`
    """
    auc_train = (roc_auc_score(y_true=y_train,
                               y_score=predictions.y_score_train)
                 .round(3))

    auc_test = (roc_auc_score(y_true=y_test,
                              y_score=predictions.y_score)
                .round(3))

    acc_train = (accuracy_score(y_true=y_train,
                                y_pred=predictions.y_pred_train)
                 .round(3))

    acc_test = (accuracy_score(y_true=y_test,
                               y_pred=predictions.y_pred))

    pr_auc_train = (average_precision_score(y_true=y_train,
                                            y_score=predictions.y_score_train)
                    .round(3))

    pr_auc_test = (average_precision_score(y_true=y_test,
                                           y_score=predictions.y_score)
                   .round(3))

    return Metrics(auc_train=auc_train, auc_test=auc_test,
                   acc_train=acc_train, acc_test=acc_test,
                   pr_auc_train=pr_auc_train, pr_auc_test=pr_auc_test)


def shap_summary_plot(shap_values: np.ndarray, dataset: pd.DataFrame,
                      max_display: int) -> None:
    """Alias for `summary_plot` method of `shap` library, adapted it
    to streamlit rendering
    """
    shap.summary_plot(shap_values,
                      dataset,
                      max_display=max_display)

    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
    plt.clf()
