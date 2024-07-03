import streamlit as st
import os
from PIL import Image
# from modeling.preprocessing import *
# from modeling.inference import *
from modeling.preprocessing_test_data import *
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
    page_title="Testing",
    page_icon="🤖",
    layout = "wide"
)

st.title("Testing")
st.markdown("### Upload a CSV file to make driving behavior predictions.")

model_path = './saving_model/lightgbm_save.pkl'
model = joblib.load(model_path)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # CSV 파일 읽기
    test_data = pd.read_csv(uploaded_file)
    X_data, y_data = preprocess_data(test_data)
    time.sleep(.9)
    st.write("Data uploaded successfully!")
    st.write("Please push the `Predict` button to make the predictions.")
    # st.write(test_data.head())
    # st.write(test_data)

    # 추론 버튼
    if st.button("Predict"):
        # 추론 수행
        X_test_data = load_data('X_test')
        predictions = model.predict(X_test_data)
        
        # 결과 데이터프레임 생성
        result_df = X_test_data.copy()
        result_df['Prediction'] = predictions

        # 예측값을 'Aggressive'와 'Normal'로 매핑
        prediction_labels = {0: 'Aggressive', 1: 'Normal'}
        result_df['Prediction'] = result_df['Prediction'].map(prediction_labels)

        result_df.to_csv('./prediction_data/result_df.csv', index=False)
        time.sleep(.9)
        st.write("Prediction was ended. Please check the prediction result on the `Results` page.")
