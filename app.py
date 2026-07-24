import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import joblib
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision.transforms import transforms
from collections import deque, Counter
from assets import load_asset

prediction_history = deque(maxlen=25)

#model & transform loading
device,label_encoder,detector,transform,model=load_asset()

st.set_page_config(
    page_title="RealTime-MaskDetection-MobileNet-MediaPipe",
    page_icon="😷",
    layout="wide"
)

#sidebar with my information+camera switch
with st.sidebar:
    st.title("⚙️ Control Panel")
    col1, col2 = st.columns(2)
    with col1:
        run_cam = st.button("▶️ START", use_container_width=True, type="primary")
    with col2:
        stop_cam = st.button("⏹️ STOP", use_container_width=True)
    st.markdown("---")