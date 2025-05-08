import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', force_reload=True)
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
