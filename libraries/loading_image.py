import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

@st.cache_data
def load_image(image_name):
    return Image.open(f'./image/{image_name}.png')