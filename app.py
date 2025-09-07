import streamlit as st
import librosa
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

st.title("🎵 Music Genre Classifier (WAV only)")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features("temp.wav")
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]  # ✅ Already a genre name

    st.success(f"🎶 Predicted Genre: **{prediction}**")











