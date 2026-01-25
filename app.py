import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="wide"
)

# ===============================
# LANGUAGE TEXT
# ===============================
texts = {
    "id": {
        "title": "Fashion Image Classifier",
        "subtitle": "Klasifikasi citra pakaian berbasis CNN (Fashion-MNIST)",
        "upload": "Unggah gambar pakaian",
        "predict": "Hasil Prediksi",
        "dataset_title": "Dataset Explorer",
        "dataset_desc": "Berikut contoh citra Fashion-MNIST (grayscale 28×28).",
        "profile": "Profil Aplikasi",
        "profile_desc": "Aplikasi klasifikasi citra pakaian menggunakan Convolutional Neural Network (CNN)."
    },
    "en": {
        "title": "Fashion Image Classifier",
        "subtitle": "CNN-based clothing image classification (Fashion-MNIST)",
        "upload": "Upload clothing image",
        "predict": "Prediction Result",
        "dataset_title": "Dataset Explorer",
        "dataset_desc": "Sample Fashion-MNIST images (grayscale 28×28).",
        "profile": "Application Profile",
        "profile_desc": "Clothing image classification application using Convolutional Neural Network (CNN)."
    }
}

# ===============================
# LANGUAGE SWITCH (FLAG)
# ===============================
if "lang" not in st.session_state:
    st.session_state.lang = "id"

col_lang1, col_lang2 = st.columns(2)
with col_lang1:
    if st.button("🇮🇩 Bahasa"):
        st.session_state.lang = "id"
with col_lang2:
    if st.button("🇬🇧 English"):
        st.session_state.lang = "en"

lang = st.session_state.lang
t = texts[lang]

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

# ===============================
# CLASS LABELS
# ===============================
class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# ===============================
# SIDEBAR MENU
# ===============================
menu = st.sidebar.radio(
    "Menu",
    ["Home", "Dataset", "Profile"]
)

# ===============================
# HOME
# ===============================
if menu == "Home":
    st.title(t["title"])
    st.subheader(t["subtitle"])

    uploaded_file = st.file_uploader(
        t["upload"],
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Input Image", width=200)

        img = image.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"{t['predict']}: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

# ===============================
# DATASET EXPLORER (NO LOCAL FOLDER)
# ===============================
elif menu == "Dataset":
    st.title(t["dataset_title"])
    st.write(t["dataset_desc"])

    sample_images = {
        "T-shirt / Top": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/0.png",
        "Trouser": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/1.png",
        "Pullover": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/2.png",
        "Dress": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/3.png",
        "Coat": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/4.png",
        "Sandal": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/5.png",
        "Shirt": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/6.png",
        "Sneaker": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/7.png",
        "Bag": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/8.png",
        "Ankle Boot": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/9.png",
    }

    selected_class = st.selectbox("Choose class", list(sample_images.keys()))
    st.image(
        sample_images[selected_class],
        caption=selected_class,
        width=200
    )

# ===============================
# PROFILE
# ===============================
elif menu == "Profile":
    st.title(t["profile"])
    st.write(t["profile_desc"])

    st.markdown("""
    **Model**: Convolutional Neural Network (CNN)  
    **Dataset**: Fashion-MNIST  
    **Image Size**: 28 × 28 (Grayscale)  
    **Classes**: 10 kategori pakaian  
    **Deployment**: Streamlit Cloud
    """)

    st.info("Aplikasi ini bersifat umum dan dapat digunakan oleh siapa saja.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("© 2026 Fashion Image Classifier | CNN + Streamlit")
