import streamlit as st
from libraries.loading_image import *

st.set_page_config(
    page_title="Information",
    page_icon="🚘",
    layout = "wide"
)

st.title("Information")
st.markdown("""
            ### Context \n
            Aggressive driving behavior is the leading factor of road traffic accidents.\n
            As reported by the AAA Foundation for Traffic Safety, 106,727 fatal crashes - 55.7 percent of the total – during a recent four-year period involved drivers who committed one or more aggressive driving actions.\n
            Therefore, how to predict dangerous driving behavior quickly and accurately?\n
            """)

st.markdown("""
            ### Solution Approach \n
            Aggressive driving includes speeding, sudden breaks and sudden left or right turns.\n
            All these events are reflected on accelerometer and gyroscope data.\n
            Therefore, knowing that almost everyone owns a smartphone nowadays which has a wide variety of sensors, a data collector application in android based on the accelerometer and gyroscope sensors have been designed.\n
            """)

st.markdown("""
            ### Content \n
            - Sampling Rate: 2 samples (rows) per second.\n
            - Gravitational acceleration: removed.\n
            - Data:\n
                a. Acceleration (X,Y,Z axis in meters per second squared (m/s2))\n
                b. Rotation (X,Y, Z axis in degrees per second (°/s))\n
                c. Classification label (Normal and Aggressive)\n
                d. Timestamp (time in seconds)\n
            - Driving behaviors:\n
                a. Normal\n
                b. Aggressive\n
            - Device: Samsung Galaxy S21\n
            """)


st.markdown("""
            ### Predictive Model \n
            """)
predictive_model = load_image('lightgbm_model')
st.image(predictive_model, width=600)

st.markdown("""
            ### MLOps Pipeline \n
            """)
mlops_flow = load_image('flow')
st.image(mlops_flow, width=600)