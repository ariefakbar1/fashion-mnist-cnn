import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fashion Image Classifier",
    layout="centered"
)

# =========================
# LANGUAGE
# =========================
lang = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Indonesia"])

TEXT = {
    "English": {
        "title": "Fashion Image Classifier",
        "upload": "Upload clothing image",
        "predict": "Prediction Result",
        "confidence": "Prediction Confidence",
        "dataset": "Dataset Preview (Fashion-MNIST)",
        "desc": "Upload an image of clothing to classify it.",
    },
    "Bahasa": {
        "title": "Klasifikasi Citra Pakaian",
        "upload": "Unggah gambar pakaian",
        "predict": "Hasil Prediksi",
        "confidence": "Tingkat Kepercayaan",
        "dataset": "Pratinjau Dataset (Fashion-MNIST)",
        "desc": "Unggah gambar pakaian untuk diklasifikasikan.",
    }
}

t = TEXT[lang]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_names = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# =========================
# TITLE
# =========================
st.title(t["title"])
st.write(t["desc"])

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    st.image(image, caption="Uploaded Image", width=200)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(t["predict"])
    st.success(class_names[predicted_class])

    st.subheader(t["confidence"])
    st.progress(int(confidence * 100))
    st.write(f"{confidence*100:.2f}%")

# =========================
# DATASET PREVIEW (FEATURE 1)
# =========================
st.markdown("---")
st.subheader(t["dataset"])

(_, _), (x_test, y_test) = fashion_mnist.load_data()

cols = st.columns(4)
for i in range(12):
    with cols[i % 4]:
        st.image(
            x_test[i],
            caption=class_names[y_test[i]],
            width=120
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("CNN-based Image Classification | Fashion-MNIST Dataset")
