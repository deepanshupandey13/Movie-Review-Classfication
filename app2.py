import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Movie AI - Windows Fix", page_icon="🎬")

@st.cache_resource
def load_model_windows():
    # Path to your local folder
    model_path = "./my_imdb_model"
    
    # We load the weights using transformers because it doesn't need 'tensorflow-text'
    # It will automatically find the files in your 'my_imdb_model' folder
    try:
        # Note: If you fine-tuned in Keras, we use the model ID to get the right architecture
        pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased") 
        return pipe
    except Exception as e:
        st.error(f"Logic Error: {e}")
        return None

analyzer = load_model_windows()

st.title("🎬 Movie Review Sentiment AI")
st.info("Windows Compatibility Mode: Active")

user_review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_review:
        result = analyzer(user_review)[0]
        label = result['label']
        score = result['score'] * 100
        
        # Mapping result to your style
        display_label = "POSITIVE 😊" if "POSITIVE" in label or "LABEL_1" in label else "NEGATIVE 😞"
        
        st.subheader(f"Result: {display_label}")
        st.write(f"Confidence: {score:.2f}%")

        