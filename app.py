import streamlit as st
import joblib
from extract_features import extract_features  # your feature extraction function
from pydub import AudioSegment
import io

st.title("Music Genre Classifier 🎵")

# Load trained model and label encoder
model = joblib.load("model.joblib")
le = joblib.load("label_encoder.joblib")

# Upload audio file
audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

if audio_file is not None:
    # Convert MP3 to WAV if needed
    if audio_file.type == "audio/mpeg":  # mp3
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_data = wav_io
    else:  # already WAV
        audio_data = audio_file

    # Display audio player
    st.audio(audio_data, format="audio/wav")

    # Extract features
    features = extract_features(audio_data)  # ensure your function can handle file-like object

    # Make prediction
    prediction = model.predict(features)

    # Convert numeric label back to genre name
    predicted_genre_name = le.inverse_transform(prediction)

    # Display predicted genre
    st.success(f"Predicted Genre: {predicted_genre_name[0]}")

    # Optional: show extracted features
    st.subheader("Extracted Features")
    st.write(features)






