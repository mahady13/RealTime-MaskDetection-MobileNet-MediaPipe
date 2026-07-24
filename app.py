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

    st.header("Developer Information")
    st.markdown(
        "***Mohiuddin Mahady*** \n \nBSc in CSE from Mymensingh Engineering College(Affiliated with Dhaka University)")
    col3, col4 = st.columns([1, 1])
    with col3:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/mohiuddin-mahady/", use_container_width=True)
    with col4:
        st.link_button("Github", 'https://www.github.com/mahady13', use_container_width=True)
    st.markdown("---")

    st.subheader("⚠️ System Limitations")
    st.info("""
        **🔹 Lighting Conditions**  
        Accuracy may drop in low-light or extreme backlights.

        **🔹 Mask Patterns & Designs**  
        Optimized for solid masks. Complex patterned masks may cause misclassifications.

        **🔹 Distance & Angle**  
        Faces too far or turned at extreme sideways might not be detected.

        **🔹 Partial Occlusion**  
        Hands,glasses or hair partially covering the face can affect confidence scores.
        """)
#app ui started
st.title("😷 Real-Time AI Face Mask Detection")
st.caption("Computer Vision pipeline built with MobileNetV2 & MediaPipe")

col_video, col_metrics = st.columns([3, 1])

with col_video:
    st.subheader("Live Video Feed")
    status_banner = st.empty()
    frame_window = st.image([])

with col_metrics:
    st.subheader("Results & Metrics")
    metric_status = st.empty()
    metric_conf = st.empty()

if run_cam:
    cap=cv2.VideoCapture(0)

    while cap.isOpened() and run_cam:
        ret,frame=cap.read()
        if not ret:
            st.error("Failed to capture video feed.")
            break

        h,w,_=frame.shape
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        detection_results=detector.detect(mp_img)

        latest_label = "Searching..."
        latest_conf = 0.0

        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.bounding_box

                xmin = max(0, bbox.origin_x)
                ymin = max(0, bbox.origin_y)
                box_w = min(w - xmin, bbox.width)
                box_h = min(h - ymin, bbox.height)