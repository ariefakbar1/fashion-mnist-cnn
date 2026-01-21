import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================= CONFIG =================
st.set_page_config(page_title="Fashion Image Classifier")

# ================= SIDEBAR =================
menu = st.sidebar.radio("Menu", ["Home", "Profile"])
language = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Indonesia"])

# ================= TEXT =================
text = {
    "English": {
        "title": "Fashion Image Classifier",
        "upload": "Upload clothing image",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "about_app": "About This Application",
        "about_app_text": "This application uses a Convolutional Neural Network (CNN) to classify clothing images from the Fashion-MNIST dataset.",
        "about_dev": "About the Developer",
        "about_dev_text": "The developer is interested in Artificial Intelligence, Deep Learning, and Computer Vision.",
        "tech": "Technologies Used"
    },
    "Bahasa Indonesia": {
        "title": "Klasifikasi Citra Pakaian",
        "upload": "Unggah gambar pakaian",
        "prediction": "Hasil Prediksi",
        "confidence": "Tingkat Keyakinan",
        "about_app": "Tentang Aplikasi",
        "about_app_text": "Aplikasi ini menggunakan Convolutional Neural Network (CNN) untuk mengklasifikasikan citra pakaian dari dataset Fashion-MNIST.",
        "about_dev": "Tentang Pengembang",
        "about_dev_text": "Pengembang memiliki minat pada Artificial Intelligence, Deep Learning, dan Computer Vision.",
        "tech": "Teknologi yang Digunakan"
    }
}

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("fashion_mnist_cnn.h5")

class_names = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# ================= HOME =================
if menu == "Home":
    st.title(text[language]["title"])

    uploaded_file = st.file_uploader(
        text[language]["upload"],
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))
        st.image(image, width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = class_names[int(np.argmax(prediction))]
        confidence = float(np.max(prediction) * 100)

        st.success(text[language]["prediction"] + ": " + predicted_class)
        st.write(text[language]["confidence"] + ": " + str(round(confidence, 2)) + "%")

# ================= PROFILE =================
if menu == "Profile":
    st.title("Profile")

    st.subheader(text[language]["about_app"])
    st.write(text[language]["about_app_text"])

    st.subheader(text[language]["about_dev"])
    st.write(text[language]["about_dev_text"])

    st.subheader(text[language]["tech"])
    st.write("Python")
    st.write("TensorFlow / Keras")
    st.write("Convolutional Neural Network (CNN)")
    st.write("Streamlit")
