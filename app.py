import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import joblib
import mediapipe as mp
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, RTCConfiguration
from PIL import Image
from torchvision.transforms import transforms
from collections import deque, Counter
from assets import load_asset

prediction_history = deque(maxlen=25)

# Model & transform loading
device, label_encoder, detector, transform, model = load_asset()

st.set_page_config(
    page_title="RealTime-MaskDetection-MobileNet-MediaPipe",
    page_icon="😷",
    layout="wide"
)

# Sidebar with developer information
with st.sidebar:
    st.title("⚙️ Control Panel")
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
        Hands, glasses or hair partially covering the face can affect confidence scores.
        """)

# App UI started
st.title("😷 Real-Time AI Face Mask Detection")
st.caption("Computer Vision pipeline built with MobileNetV2 & MediaPipe")


# Callback function for processing each frame continuously in WebRTC
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_results = detector.detect(mp_img)

    if detection_results.detections:
        for detection in detection_results.detections:
            bbox = detection.bounding_box

            xmin = max(0, bbox.origin_x)
            ymin = max(0, bbox.origin_y)
            box_w = min(w - xmin, bbox.width)
            box_h = min(h - ymin, bbox.height)

            if box_w > 0 and box_h > 0:
                crop_ymin = ymin
                crop_ymax = min(h, ymin + int(box_h * 0.90))

                face_crop = rgb_frame[crop_ymin:crop_ymax, xmin:xmin + box_w]
                pil_img = Image.fromarray(face_crop)
                tensor_input = transform(pil_img).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output = model(tensor_input)
                    probabilities = torch.softmax(output, dim=1)
                    conf, pred_idx = torch.max(probabilities, dim=1)
                    confidence = conf.item()
                    pred = pred_idx.item()

                raw_label = label_encoder.inverse_transform([pred])[0]
                if raw_label == 'with_mask' and confidence < 0.80:
                    raw_label = 'without_mask'
                prediction_history.append(raw_label)
                smoothed_label = Counter(prediction_history).most_common(1)[0][0]

                color = (0, 255, 0) if smoothed_label == 'with_mask' else (0, 0, 255)
                text = 'Mask Found' if smoothed_label == 'with_mask' else 'Mask Not Found'
                display_text = f"{text} ({confidence * 100:.1f}%)"

                cv2.rectangle(img, (xmin, ymin), (xmin + box_w, ymin + box_h), color, 3)
                cv2.putText(
                    img, display_text, (xmin, max(20, ymin - 10)),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2
                )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


col_video, col_metrics = st.columns([3, 1])

with col_video:
    st.subheader("Live Video Feed")
    webrtc_streamer(
        key="mask-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )