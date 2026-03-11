import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load model and tokenizer with error handling ---
try:
    model = load_model("bilstm_sentiment_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

# Set max length same as used in training
MAX_LENGTH = 200  # adjust if different during training

# --- Streamlit UI ---
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")
st.title("IMDB Review Sentiment Analysis")
st.write("Enter a movie review below and find out if it's positive or negative!")

# User input
user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review!")
    else:
        try:
            # Tokenize and pad
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

            # Make prediction
            pred = model.predict(padded)[0][0]

            # Convert prediction to label
            if pred >= 0.5:
                st.success(f"Positive Review 😄 (Confidence: {pred:.2f})")
            else:
                st.error(f"Negative Review 😟 (Confidence: {1 - pred:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


#export PATH=$PATH:/c/Users/DEEPANSHU/AppData/Roaming/Python/Python313/Scripts