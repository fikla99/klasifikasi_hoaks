
import streamlit as st
import pickle
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# CSS untuk mempercantik tampilan
st.markdown("""
    <style>
        .title {
            font-size:36px !important;
            color:#4CAF50;
            font-weight: bold;
        }
        .sub-title {
            font-size:18px;
            color:#555;
            margin-bottom: 30px;
        }
        .footer {
            font-size:14px;
            margin-top: 30px;
            color: #999;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown('<div class="title">🔍 Deteksi Berita HOAKS atau FAKTA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Gunakan AI untuk membantu mendeteksi kebenaran informasi yang Anda baca.</div>', unsafe_allow_html=True)

# Load model dan vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Stopword remover
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

# Input form
st.markdown("### 📝 Masukkan Informasi Berita")
headline = st.text_input("📌 Judul Berita:")
body = st.text_area("📰 Isi Berita:")

# Tombol prediksi
if st.button("🔎 Prediksi"):
    if not headline and not body:
        st.warning("Masukkan headline atau body berita terlebih dahulu.")
    else:
        text = headline + " " + body
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == "FAKTA":
            st.success("✅ Ini adalah FAKTA.")
        else:
            st.error("🚫 Ini terindikasi sebagai HOAKS.")

# Footer
st.markdown('<div class="footer">Dibuat dengan ❤️ menggunakan Streamlit, Sastrawi, dan Scikit-Learn</div>', unsafe_allow_html=True)
