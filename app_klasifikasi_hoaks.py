
import streamlit as st
import pickle
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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

# UI Streamlit
st.title("Deteksi Berita HOAKS / FAKTA")
st.write("Masukkan teks berita untuk mengetahui apakah itu HOAKS atau FAKTA.")

headline = st.text_input("Judul Berita (Headline):")
body = st.text_area("Isi Berita (Body):")

if st.button("Prediksi"):
    if headline.strip() == "" and body.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        text = headline + " " + body
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Hasil Prediksi: {prediction}")
