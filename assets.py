import torch
import joblib
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st

@st.cache_resource
def load_asset():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 2),
    )
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    label_encoder=joblib.load("label_encoder.pkl")

    base_options = python.BaseOptions(model_asset_path='face_detector.tflite')
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    detector = vision.FaceDetector.create_from_options(options=options)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return device,label_encoder,detector,transform,model