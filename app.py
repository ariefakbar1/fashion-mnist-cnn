import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered"
)

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

# =========================
# LANGUAGE
# =========================
lang = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Bahasa Indonesia"])

TEXT = {
    "title": "Fashion Image Classifier" if lang == "English" else "Klasifikasi Gambar Fashion",
    "upload": "Upload clothing image" if lang == "English" else "Upload gambar pakaian",
    "dataset": "Dataset Explorer" if lang == "English" else "Eksplorasi Dataset",
    "prediction": "Prediction Result" if lang == "English" else "Hasil Prediksi",
    "confidence": "Prediction Confidence" if lang == "English" else "Tingkat Keyakinan Model"
}

# =========================
# HEADER
# =========================
st.title(TEXT["title"])
st.write("CNN-based image classification using Fashion-MNIST")

menu = st.sidebar.radio("Menu", ["Home", "Dataset", "Profile"])

# =========================
# HOME
# =========================
if menu == "Home":
    uploaded_file = st.file_uploader(
        TEXT["upload"],
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array)[0]

        # Top 3 Prediction
        top3_idx = predictions.argsort()[-3:][::-1]

        st.subheader(TEXT["prediction"])
        for i in top3_idx:
            st.write(f"**{CLASS_NAMES[i]}** : {predictions[i]*100:.2f}%")

        # Confidence Bar Chart
        st.subheader(TEXT["confidence"])
        chart_data = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in top3_idx
        }
        st.bar_chart(chart_data)

# =========================
# DATASET EXPLORER
# =========================
elif menu == "Dataset":
    st.subheader(TEXT["dataset"])

    dataset_path = "dataset"
    selected_class = st.selectbox("Choose Class", CLASS_NAMES)

    class_path = os.path.join(dataset_path, selected_class)

    if os.path.exists(class_path):
        images = os.listdir(class_path)
        sample_images = random.sample(images, min(6, len(images)))

        cols = st.columns(3)
        for idx, img_name in enumerate(sample_images):
            img = Image.open(os.path.join(class_path, img_name))
            cols[idx % 3].image(img, caption=selected_class, use_column_width=True)
    else:
        st.warning("Dataset folder not found!")

# =========================
# PROFILE
# =========================
elif menu == "Profile":
    st.subheader("👤 Developer Profile")

    st.write("**Name:** Muhammad Arief Akbar")
    st.write("**Project:** Fashion Image Classification using CNN")
    st.write("**Technology:** Deep Learning, CNN, TensorFlow, Streamlit")
    st.write("**Dataset:** Fashion-MNIST (Image Version)")
