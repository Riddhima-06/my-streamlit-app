import streamlit as st
import numpy as np
import pandas as pd
import joblib
from extract_features import extract_features

st.title("🎵 Music Genre Classifier")

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")   # make sure this file is in your repo

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Extract features
    features = extract_features(uploaded_file)
    if features is not None:
        features = np.array(features).reshape(1, -1)

        # Predict genre
        prediction = model.predict(features)[0]

        st.success(f"🎶 Predicted Genre: **{prediction}**")

        # Show extracted features for reference
        df = pd.DataFrame([features[0]])
        st.write("🔍 Extracted Features:")
        st.dataframe(df)
    else:
        st.error("❌ Could not extract features from the file.")





