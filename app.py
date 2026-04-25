import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model
model = load_model("rnn_model.h5")

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 25

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Convert to sequence
def text_to_seq(text):
    words = text.split()
    return [vocab.get(word, vocab['<OOV>']) for word in words]

# Preprocess
def preprocess(text):
    text = clean_text(text)
    seq = text_to_seq(text)
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    return padded

# UI
st.title("💬 Hate Speech Detection (RNN)")
st.write("Enter a tweet to classify")

user_input = st.text_area("Enter Tweet")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        processed = preprocess(user_input)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.error(f"⚠️ Hate Speech (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ Not Hate Speech (Confidence: {prediction:.2f})")