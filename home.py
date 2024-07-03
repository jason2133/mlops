# streamlit run home.py --client.showErrorDetails=false

import streamlit as st
import os
from PIL import Image
from modeling.preprocessing import *
from modeling.inference import *
from libraries.check_predict_output import *
from libraries.loading_image import *
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="home",
    page_icon="🏠"
)

st.header('Driving Behavior Prediction using MLOps')
st.markdown("""
            Developed by [Jaeseung Lee](https://github.com/jason2133)\n
            School of Electrical Engineering, Korea University
            """)

driving_image = load_image('driving_image')
st.image(driving_image, width=600)

st.write(
    """
    This is the dashboard for driving behavior prediction.\n
    The results may be classified as Normal or Aggressive.\n
    On the `Testing` page, you can conduct a test for predicting driving behavior.\n
    On the `Results` page, you can check the results of the driving behavior prediction.
    """
)

if check_predict_output_score():
    os.remove('./prediction_data/result_df.csv')

try:
    # Sidebar
    st.page_link("home.py", label="Home")
    st.page_link("pages/1_information.py", label="🚘 Information")
    st.page_link("pages/2_testing.py", label="🤖 Testing")
    st.page_link("pages/3_results.py", label="🎯 Results")
    # st.markdown("---")
except:
    pass