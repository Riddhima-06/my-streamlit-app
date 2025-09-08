import os
import joblib
import streamlit as st
from extract_features import main as features_main
from train import main as train_main

# Step 1: Ensure features.csv exists
if not os.path.exists("features.csv"):
    st.warning("⚠️ No features dataset found. Extracting now...")
    features_main()
    st.success("✅ Features extracted and saved to features.csv!")

# Step 2: Check if model & scaler exist, else train
if not os.path.exists("model.joblib") or not os.path.exists("scaler.joblib"):
    st.warning("⚠️ No trained model found. Training now, please wait...")
    train_main()
    st.success("✅ Model trained and saved successfully!")

# Step 3: Load model & scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# 🎵 Define genre labels (adjust this order to match your dataset)
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop", 
    "jazz", "metal", "pop", "reggae", "rock"
]

# Example placeholder for your feature extraction
def extract_features(file_path):
    # Call your real feature extraction logic from extract_features.py
    # It should return a numpy array of features
    pass

# Prediction function
def predict(file_path):
    features = extract_features(file_path)
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    predicted_genre = GENRE_LABELS[prediction[0]]  # Map numeric label → genre
    return predicted_genre

# Streamlit UI
st.title("🎵 Music Genre Classifier")

uploaded_file = st.file_uploader("Upload a music file (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    prediction = predict("temp.wav")
    st.success(f"🎶 Predicted Genre: **{prediction.capitalize()}**")


