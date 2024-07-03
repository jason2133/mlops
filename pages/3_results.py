import streamlit as st
import os
from PIL import Image
# from modeling.preprocessing import *
# from modeling.inference import *
from libraries.check_predict_output import *
from libraries.loading_data import *
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Results",
    page_icon="🎯",
    layout = "wide"
)

st.title("Result Analysis")

if not check_predict_output_score():
    st.warning("Please test the data on the `Testing` page")
else:
    result_df = load_prediction_data('result_df')

    st.write("### Predictions:")
    st.write(result_df)

    st.write("""
            ### The change in driving behavior over time: \n
             Upper 50 data.
             """)
    fig, ax = plt.subplots(figsize=(6, 4))
    # Prediction
    prediction_value = result_df['Prediction'].values.tolist()
    prediction_value_encoding = [0 if behavior == 'Normal' else 1 for behavior in prediction_value]
    time_length = np.arange(len(prediction_value_encoding))
    ax.plot(time_length[:50], prediction_value_encoding[:50], linestyle='-', marker='o', markersize=2, label='Driving Behavior')
    ax.set_xlabel('Time')
    ax.set_ylabel('Driving Behavior')
    # ax.set_title('Change in Driving Behavior Over Time')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Aggressive'])
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    # 레이아웃 조정 및 그래프 출력
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Prediction Distribution:")
    fig, ax = plt.subplots(figsize=(6, 4))
    result_df['Prediction'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig)


