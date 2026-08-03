import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection using VGG16")

MODEL_PATH = "cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img)

    probability = float(prediction[0][0])

    st.write(f"Prediction Score: {probability:.4f}")

    if probability >= 0.5:
        st.error("🦠 Pneumonia Detected")
    else:
        st.success("✅ Normal Chest X-ray")