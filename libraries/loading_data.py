import os
import streamlit as st
import numpy as np
import pandas as pd

@st.cache_data
def load_data(data):
    file_dir = f'./dataset/{data}.csv'
    return pd.read_csv(file_dir)

@st.cache_data
def load_prediction_data(data):
    file_dir = f'./prediction_data/{data}.csv'
    return pd.read_csv(file_dir)
