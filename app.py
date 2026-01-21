import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    menu = st.sidebar.radio(
    "Menu",
    ["Home", "Profile"]
    )
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered"
)

# =============================
# LANGUAGE SELECTION
# =============================
language = st.sidebar.selectbox(
    "🌐 Language / Bahasa",
    ("English", "Bahasa Indonesia")
)

# =============================
# TEXT DICTIONARY
# =============================
text = {
    "English": {
        "title": "👕 Fashion Image Classifier",
        "subtitle": "Upload a clothing image and get an instant prediction using deep learning",
        "about": "ℹ️ About This App",
        "about_text": """
This web application uses a **Convolutional Neural Network (CNN)**  
to classify clothing images into **10 categories**.
""",
        "upload": "📤 Upload clothing image",
        "uploaded": "Uploaded Image",
        "prediction": "Prediction",
        "confidence": "Confidence"
    },
    "Bahasa Indonesia": {
        "title": "👕 Klasifikasi Citra Pakaian",
        "subtitle": "Unggah gambar pakaian dan dapatkan prediksi menggunakan deep learning",
        "about": "ℹ️ Tentang Aplikasi",
        "about_text": """
Aplikasi web ini menggunakan **Convolutional Neural Network (CNN)**  
untuk mengklasifikasikan gambar pakaian ke dalam **10 kategori**.
""",
        "upload": "📤 Unggah gambar pakaian",
        "uploaded": "Gambar yang diunggah",
        "prediction": "Hasil Prediksi",
        "confidence": "Tingkat Keyakinan"
    }
}

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
st.markdown(f"<div class='title'>{text[language]['title']}</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='subtitle'>{text[language]['subtitle']}</div>",
    unsafe_allow_html=True
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title(text[language]["about"])
st.sidebar.markdown(text[language]["about_text"])

# =============================
# UPLOAD IMAGE
# =============================
if menu == "Home":
uploaded_file = st.file_uploader(
    text[language]["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption=text[language]["uploaded"], width=150)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(
        f"<div class='prediction'>{text[language]['prediction']}: {predicted_class}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p>{text[language]['confidence']}: <b>{confidence:.2f}%</b></p>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
if menu == "Profile":
    st.title("Profile")

    st.write("### About This Application")
    st.write(
        "This web application demonstrates the use of a Convolutional Neural Network (CNN) "
        "for classifying clothing images using the Fashion-MNIST dataset."
    )

    st.write("### About the Developer")
    st.write(
        "The developer is interested in Artificial Intelligence, Deep Learning, "
        "and Computer Vision, with a focus on practical applications."
    )

    st.write("### Technologies Used")
    st.write("- Python")
    st.write("- TensorFlow / Keras")
    st.write("- Convolutional Neural Network (CNN)")
    st.write("- Streamlit")
