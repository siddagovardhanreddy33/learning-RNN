import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------
# Load model + tokenizer
# ----------------------
model = load_model("model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 300  # same as training

# ----------------------
# UI
# ----------------------
st.title("🧠 Sentiment Analysis (RNN)")
st.write("Enter text and check if it's Positive or Negative")

user_input = st.text_area("Enter your text here:")

# ----------------------
# Prediction function
# ----------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    
    prediction = model.predict(padded)[0][0]
    
    if prediction > 0.5:
        return "😊 Positive", prediction
    else:
        return "😡 Negative", prediction

# ----------------------
# Button
# ----------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, score = predict_sentiment(user_input)
        
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {score:.4f}")