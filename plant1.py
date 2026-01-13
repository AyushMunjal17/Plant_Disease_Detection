import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="centered",
)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

MODEL_PATH = Path("trained_model.keras")
IMAGE_SIZE = (128, 128)


def resolve_model_url() -> str | None:
    if "MODEL_URL" in st.secrets:
        return st.secrets["MODEL_URL"]
    return os.getenv("MODEL_URL")


def download_model_if_needed() -> None:
    if MODEL_PATH.exists():
        return

    model_url = resolve_model_url()
    if not model_url:
        raise FileNotFoundError(
            f"Trained model file not found at {MODEL_PATH.resolve()} and "
            "no MODEL_URL secret/environment variable provided."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(model_url, timeout=60, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model() -> tf.keras.Model:
    download_model_if_needed()
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr


def model_prediction(image_bytes: bytes) -> Tuple[str, float]:
    model = load_model()
    input_arr = prepare_image(image_bytes)
    predictions = model.predict(input_arr, verbose=0)
    result_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return CLASS_NAMES[result_index], confidence


def render_home():
    st.header("🌿 Plant Disease Recognition System")
    st.markdown(
        """
        Upload a plant leaf image to detect common diseases using a TensorFlow model
        trained on the PlantVillage dataset.

        **How it works**
        1. The image is resized to 128×128 pixels.
        2. The TensorFlow CNN model classifies it into one of 38 classes.
        3. You get the predicted disease and confidence score instantly.
        """
    )


def render_inference():
    st.header("🌿 Plant Disease Recognition")
    uploaded_file = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    else:
        st.info("Please upload a clear photo of a single leaf to begin.")

    if st.button("Predict", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("Upload an image before predicting.")
            return
        with st.spinner("Analyzing image..."):
            try:
                label, confidence = model_prediction(uploaded_file.getvalue())
                st.success(f"Prediction: **{label}** ({confidence:.1%} confidence)")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to make a prediction: {exc}")


def main():
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox(
        "Select Page", ["Home", "Plant Disease Recognition"]
    )

    if app_mode == "Home":
        render_home()
    else:
        render_inference()


if __name__ == "__main__":
    main()
