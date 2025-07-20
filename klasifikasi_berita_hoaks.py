
import pandas as pd
import numpy as np
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load dataset
df = pd.read_csv("Scrapping.csv", encoding="latin-1", sep=';', on_bad_lines='skip')

# Ambil kolom yang diperlukan dan buang yang kosong
df = df[["Headline", "Body", "Label"]].dropna()

# Filter hanya label yang relevan
df = df[df["Label"].isin(["Fakta", "Disinformasi", "DISINFORMASI", "HOAKS", "fitnah"])]

# Normalisasi label
df["Label"] = df["Label"].astype(str).str.upper()
df["Label"] = df["Label"].replace({
    "DISINFORMASI": "HOAKS",
    "FITNAH": "HOAKS"
})

# Gabungkan Headline dan Body ke dalam satu kolom text
df["text"] = df["Headline"] + " " + df["Body"]

# Ambil hanya kolom yang akan digunakan dan ubah nama Label -> label
df = df[["text", "Label"]].rename(columns={"Label": "label"})

# Stopword removal
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# Fitur dan label
X = df["clean_text"]
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Training model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model dan vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model dan vectorizer berhasil disimpan.")
