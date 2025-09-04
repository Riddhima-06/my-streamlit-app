import streamlit as st
import joblib
from src.feature_extraction import extract_features
import numpy as np

# Load model
clf = joblib.load("genre_classifier.joblib")

# Streamlit UI
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file and let the model predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "au", "mp3"])

if uploaded_file is not None:
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        # Extract features
        feats = extract_features("temp_audio")
        # Predict
        probs = clf.predict_proba([feats])[0]
        pred = clf.classes_[np.argmax(probs)]
        
        st.success(f"**Predicted Genre:** {pred}")
        
        st.write("### Probabilities:")
        for genre, p in zip(clf.classes_, probs):
            st.write(f"{genre}: {p:.2f}")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
