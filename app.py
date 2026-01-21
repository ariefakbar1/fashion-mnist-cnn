import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fashion Image Classifier")

# MENU
menu = st.sidebar.radio("Menu", ["Home", "Profile"])

# LOAD MODEL
model = tf.keras.models.load_model("fashion_mnist_cnn.h5")

class_names = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# ================= HOME =================
if menu == "Home":
    st.title("Fashion Image Classifier")

    uploaded_file = st.file_uploader(
        "Upload clothing image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))
        st.image(image, caption="Uploaded Image", width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

# ================= PROFILE =================
if menu == "Profile":
    st.title("Profile")

    st.subheader("About This Application")
    st.write(
        "This web application demonstrates the use of a Convolutional Neural Network (CNN) "
        "for classifying clothing images using the Fashion-MNIST dataset."
    )

    st.subheader("About the Developer")
    st.write(
        "The developer is interested in Artificial Intelligence, Deep Learning, "
        "and Computer Vision, focusing on practical applications."
    )

    st.subheader("Technologies Used")
    st.write("- Python")
    st.write("- TensorFlow / Keras")
    st.write("- Convolutional Neural Network (CNN)")
    st.write("- Streamlit")
