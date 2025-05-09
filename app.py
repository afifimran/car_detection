import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile
import sys
from pathlib import Path
@st.cache_resource
def load_model():
    # Add YOLOv5 repo to sys.path
    yolo_path = Path("yolov5")
    sys.path.append(str(yolo_path.resolve()))

    # Load custom model from local repo
    model = torch.hub.load(
        str(yolo_path),  # Local repo path
        'custom',        # Custom model loading
        path='best.pt',  # Path to your weights
        source='local'   # Important: tells torch to load from local path
    )
    return model

model = load_model()

st.title("Car Detection App")
st.write("Upload an image to detect cars and show bounding boxes with manufacturer names (if available).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

    # Render results on image
    results.render()  # updates results.ims with boxes and labels

    # Display result image
    st.image(results.ims[0], caption="Detection Results", use_container_width=True)

    # Show detected labels
    st.subheader("Detected Objects:")
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        st.write(f"{label} ({conf:.2f})")
