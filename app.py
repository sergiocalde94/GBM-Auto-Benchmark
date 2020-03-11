import streamlit as st
import pandas as pd
import re
import configparser
import shap
import category_encoders as ce

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from catboost import (CatBoostClassifier,
                      CatboostError,
                      Pool)
from lightgbm import LGBMClassifier

from helpers.utils import read_dataset
from helpers.models import (get_predictions,
                            compute_metrics,
                            convert_to_category_dtype,
                            fit_model,
                            shap_summary_plot)


config = configparser.ConfigParser()

upload_file = st.checkbox(
    'Do you want to upload your own config.ini? '
    '(if not titanic example in repo will be used)'
)

if upload_file:
    st.markdown(
        'It´s compulsory to contain **`DATA`** and **`MODEL`** sections'
    )

    st.text('''
    - DATA
        - path
        - sep
        - name
        - target
        - na_values
    - MODEL
        - test_size_proportion
        - random_seed''')

    st.text('You can pass url instead a local path in you config file')
    st.text('If you are on Heroku you can only use url or default config')
    st.text('If you cloned the repo you can use any dataset you got in local')
    file = st.file_uploader(label='', type='ini')
    config.read_string(file.getvalue())
else:
    config.read('config.ini')

PATH_PROJECT = Path('.')

URL_PATTERN = (r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}'
               r'\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)')

IS_URL = re.match(URL_PATTERN, config['DATA']['path'])

DATASET_PATH = (PATH_PROJECT / config['DATA']['path'] if not IS_URL
                else config['DATA']['path'])
DATASET_SEP = config['DATA']['sep']
DATASET_NAME = config['DATA']['name'].capitalize()
DATASET_TARGET = config['DATA']['target']
DATASET_NA_VALUES = config['DATA']['na_values']

TEST_SIZE = config['MODEL'].getfloat('test_size_proportion')
RANDOM_SEED = config['MODEL'].getint('random_seed')

df = read_dataset(DATASET_PATH,
                  sep=DATASET_SEP,
                  na_values=DATASET_NA_VALUES,
                  dtype={DATASET_TARGET: str})

if df.empty:
    st.error('Check your config DATA path variable!')
else:
    st.title(f'Welcome to GBM Auto Benchmark :nerd_face:')

    st.text('All the parameters are set to its default form')

    st.text(
        'All of the NA values in numerical columns are filled with the mean'
    )

    st.text(
        'All of the NA values in categorical columns are filled with the mode'
    )

    st.markdown(f'### Sample of the {DATASET_NAME} data')
    st.dataframe(
        df
        .sample(min(10, len(df)))
        .reset_index(drop=True)
        .style
        .highlight_null(null_color='gray')
    )

    number_rows, number_columns = df.shape
    st.write('Your data has',
             number_rows, 'rows and',
             number_columns, 'columns')

    if number_rows < 1000:
        st.warning(
            'Are you sure you want to test GBM models?, your data is small, '
            'logistic regression exists too :joy:'
        )

    columns_numerical = df.select_dtypes(include='number').columns
    columns_category = (
        df
        .select_dtypes(include='object')
        .drop(columns=DATASET_TARGET).columns
    )

    option_numerical = st.multiselect(
        'Select numerical features you want to include in the model',
        columns_numerical
    )

    option_category = st.multiselect(
        'Select categorical features you want to include in the model',
        columns_category
    )

    st.write('### Numerical Features', option_numerical)
    st.write('### Categorical Features', option_category)

    X = df[option_numerical + option_category]
    y = df[DATASET_TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    imp_numerical = SimpleImputer(strategy='median')
    imp_categorical = SimpleImputer(strategy='most_frequent')

    enc = ce.OneHotEncoder(cols=option_category,
                           use_cat_names=True,
                           handle_unknown='indicator')

    ct = ColumnTransformer(
        [("imp_numerical", imp_numerical, option_numerical),
         ("imp_categorical", imp_categorical, option_category)]
    )

    xgb = XGBClassifier()
    catboost = CatBoostClassifier(cat_features=option_category)
    lightgbm = LGBMClassifier()

    models = dict(xgb=xgb, catboost=catboost, lightgbm=lightgbm)

    try:
        X_train_imputed = (
            pd.DataFrame(ct.fit_transform(X_train),
                         columns=option_numerical + option_category)
            .astype({column: float for column in option_numerical})
            .astype({column: str for column in option_category})
        )

        X_test_imputed = (
            pd.DataFrame(ct.transform(X_test),
                         columns=option_numerical + option_category)
            .astype({column: float for column in option_numerical})
            .astype({column: str for column in option_category})
        )

        X_train_imputed_encoded = enc.fit_transform(X_train_imputed)
        X_test_imputed_encoded = enc.transform(X_test_imputed)

        models_fitted = dict(
            xgb=fit_model(models['xgb'],
                          X_train_imputed_encoded,
                          y_train),
            catboost=fit_model(models['catboost'],
                               X_train_imputed,
                               y_train),
            lightgbm=fit_model(models['lightgbm'],
                               X_train_imputed,
                               y_train,
                               is_lightgbm=True)
        )

        predictions = dict(
            xgb=get_predictions(models_fitted['xgb'],
                                X_train_imputed_encoded,
                                X_test_imputed_encoded),
            catboost=get_predictions(models_fitted['catboost'],
                                     X_train_imputed,
                                     X_test_imputed),
            lightgbm=get_predictions(models_fitted['lightgbm'],
                                     X_train_imputed,
                                     X_test_imputed,
                                     is_lightgbm=True)
        )

        metrics = dict(
            xgb=compute_metrics(predictions=predictions['xgb'],
                                y_train=y_train,
                                y_test=y_test),
            catboost=compute_metrics(predictions=predictions['lightgbm'],
                                     y_train=y_train,
                                     y_test=y_test),
            lightgbm=compute_metrics(predictions=predictions['catboost'],
                                     y_train=y_train,
                                     y_test=y_test)
        )

        st.markdown('## Models metrics :speak_no_evil:')
        st.table(pd.DataFrame(metrics, index=['auc_train', 'auc_test',
                                              'acc_train', 'acc_test',
                                              'pr_auc_train', 'pr_auc_test']))

        metric_to_use = st.radio(
            'Which metric do you want to graph?',
            ('Accuracy', 'AUC', 'PR_AUC')
        )

        metrics_translate = dict(AUC='auc', Accuracy='acc', PR_AUC='pr_auc')

        df_metrics = pd.DataFrame(metrics,
                                  index=['auc_train', 'auc_test',
                                         'acc_train', 'acc_test',
                                         'pr_auc_train', 'pr_auc_test'])

        st.line_chart(df_metrics
                      .loc[[f'{metrics_translate[metric_to_use]}_train',
                            f'{metrics_translate[metric_to_use]}_test']])

        st.markdown('## Let´s interpret the results :cop:')
        max_value = len(X_train_imputed_encoded.columns)

        num_features_to_show = (
            st.slider('How much important features you want to zoom in?',
                      min_value=1,
                      max_value=min(max_value + 1, 20),
                      value=min(5, max_value))
        )

        explainer_xgb = shap.TreeExplainer(models_fitted['xgb'])
        shap_values_xgb = explainer_xgb.shap_values(X_train_imputed_encoded)

        st.markdown('### Shap Values XGBoost')
        shap_summary_plot(shap_values_xgb,
                          dataset=X_train_imputed_encoded,
                          max_display=num_features_to_show)

        explainer_catboost = shap.TreeExplainer(models_fitted['catboost'])

        shap_values_catboost = (
            explainer_catboost
            .shap_values(Pool(X_train_imputed,
                              y_train,
                              cat_features=option_category))
        )

        st.markdown('### Shap Values CatBoost')
        shap_summary_plot(shap_values_catboost,
                          dataset=X_train_imputed,
                          max_display=num_features_to_show)

        explainer_lightgbm = shap.TreeExplainer(models_fitted['lightgbm'])

        shap_values_lightgbm = (
            explainer_lightgbm
            .shap_values(convert_to_category_dtype(X_train_imputed))
        )[1]

        st.markdown('### Shap Values LightGBM')
        shap_summary_plot(shap_values_lightgbm,
                          dataset=convert_to_category_dtype(X_train_imputed),
                          max_display=num_features_to_show)

    except (XGBoostError, CatboostError, ValueError) as exception:
        st.warning('Waiting for the features :grimacing:')
        st.error('If you have the variables added report a bug because this is '
                 'one of them :warning:')

        if st.checkbox('Show exception message (only for advanced users)'):
            st.error(str(exception))
