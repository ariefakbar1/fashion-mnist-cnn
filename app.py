import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
    text-align: center;
}
.prediction {
    font-size: 24px;
    font-weight: bold;
    color: #2c7be5;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model("fashion_mnist_cnn.h5")

class_names = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# =============================
# HEADER
# =============================
st.markdown("<div class='title'>👕 Fashion Image Classifier</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload an image of clothing and get an instant prediction using deep learning</div>",
    unsafe_allow_html=True
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This web application uses a **Convolutional Neural Network (CNN)**  
to classify clothing images into **10 categories**.

**Supported images:** JPG, PNG  
**Image size:** Automatically resized  
""")

# =============================
# UPLOAD IMAGE
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(
        f"<div class='prediction'>Prediction: {predicted_class}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p>Confidence: <b>{confidence:.2f}%</b></p>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
