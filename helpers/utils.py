import streamlit as st
import pandas as pd


@st.cache
def read_dataset(filepath_or_buffer: str, sep: str,
                 na_values: [str], dtype: [str]) -> pd.DataFrame:
    """Alias for `pd.read_csv` function and implements streamlit
    caching so we haven't to load the data all of the times that
    we rerun
    """
    try:
        return pd.read_csv(filepath_or_buffer,
                           sep=sep,
                           na_values=na_values,
                           dtype=dtype)
    except FileNotFoundError:
        return pd.DataFrame()
