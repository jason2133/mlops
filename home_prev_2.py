import streamlit as st
import os
from PIL import Image
from modeling.preprocessing import *
from modeling.inference import *
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.header('Driving Behavior Prediction using MLOps')
st.markdown("""
            Developed by [Jaeseung Lee](https://github.com/jason2133)\n
            School of Electrical Engineering, Korea University
            """)

@st.cache_data
def load_image(image_name):
    return Image.open(f'./image/{image_name}.jpg')

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

st.markdown("### Upload a CSV file to make driving behavior predictions.")

model_path = './saving_model/lightgbm_save.pkl'
model = joblib.load(model_path)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # CSV 파일 읽기
    test_data = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfully!")
    # st.write(test_data.head())
    st.write(test_data)

    # 추론 버튼
    if st.button("Predict"):
        # 추론 수행
        predictions = model.predict(test_data)
        
        # 결과 데이터프레임 생성
        result_df = test_data.copy()
        result_df['Prediction'] = predictions

        # 예측값을 'Aggressive'와 'Normal'로 매핑
        prediction_labels = {0: 'Aggressive', 1: 'Normal'}
        result_df['Prediction'] = result_df['Prediction'].map(prediction_labels)

        st.write("Predictions:")
        # st.write(result_df.head())
        st.write(result_df)
        
        # 결과 시각화
        st.write("Prediction Distribution:")

        fig, ax = plt.subplots()
        result_df['Prediction'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Count')
        st.pyplot(fig)






