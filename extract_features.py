import numpy as np
import librosa

# Function to extract features from an uploaded audio file
def extract_features(file):
    try:
        # Load audio file from uploaded file (Streamlit uploader gives a BytesIO object)
        y, sr = librosa.load(file, sr=None, duration=30)  # load first 30 seconds
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # extract 20 MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0)  # take mean of each MFCC
        return mfccs_mean
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
