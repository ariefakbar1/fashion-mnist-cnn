import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered"
)

# =========================
# LANGUAGE STATE
# =========================
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

def set_lang(lang):
    st.session_state.lang = lang

# =========================
# TEXT DICTIONARY
# =========================
TEXT = {
    "EN": {
        "title": "Fashion Image Classifier",
        "desc": "CNN-based clothing image classification using Fashion-MNIST",
        "upload": "Upload clothing image",
        "result": "Prediction Result",
        "confidence": "Prediction Confidence",
        "dataset": "Dataset Explorer",
        "dataset_desc": "Official Fashion-MNIST image samples",
        "profile": "Profile",
    },
    "ID": {
        "title": "Klasifikasi Citra Fashion",
        "desc": "Klasifikasi citra pakaian berbasis CNN menggunakan Fashion-MNIST",
        "upload": "Unggah gambar pakaian",
        "result": "Hasil Prediksi",
        "confidence": "Tingkat Keyakinan",
        "dataset": "Eksplorasi Dataset",
        "dataset_desc": "Contoh citra resmi dataset Fashion-MNIST",
        "profile": "Profil",
    }
}

T = TEXT[st.session_state.lang]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset", "Profile"]
)

st.sidebar.markdown("### Language")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("🇬🇧", on_click=set_lang, args=("EN",))
with col2:
    st.button("🇮🇩", on_click=set_lang, args=("ID",))

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

CLASS_NAMES = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# =========================
# HOME
# =========================
if menu == "Home":
    st.title(T["title"])
    st.write(T["desc"])

    uploaded_file = st.file_uploader(
        T["upload"],
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=220)

        img = image.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = preds[idx] * 100

        st.subheader(T["result"])
        st.success(label)

        st.subheader(T["confidence"])
        st.progress(int(confidence))
        st.write(f"**{confidence:.2f}%**")

# =========================
# DATASET PAGE
# =========================
elif menu == "Dataset":
    st.title(T["dataset"])
    st.write(T["dataset_desc"])

    st.markdown(
        "🔗 **Official Source:** "
        "[Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)"
    )

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

    selected = st.selectbox("Choose Class", list(sample_images.keys()))
    st.image(sample_images[selected], caption=selected, width=200)

    st.info("These are **official visual samples** from Fashion-MNIST.")

# =========================
# PROFILE
# =========================
elif menu == "Profile":
    st.title(T["profile"])

    st.markdown("""
    **Name:** Muhammad Arief Akbar  
    **Field:** Informatics – Artificial Intelligence  
    **Model:** Convolutional Neural Network (CNN)  
    **Dataset:** Fashion-MNIST (Zalando Research)  
    **Deployment:** Streamlit Cloud
    """)

    st.success("✔ Application ready for academic presentation")
