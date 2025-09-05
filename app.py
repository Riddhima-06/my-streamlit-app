import streamlit as st
import numpy as np
import pandas as pd
from extract_features import extract_features

st.title("🎵 Music Genre Feature Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Extract features
    features = extract_features(uploaded_file)
    if features is not None:
        df = pd.DataFrame([features])
        st.write("✅ Extracted Features:")
        st.dataframe(df)
    else:
        st.error("❌ Could not extract features from the file.")



