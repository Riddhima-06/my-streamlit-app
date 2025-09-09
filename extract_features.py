import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "data/genres/"
OUTPUT_CSV = "features.csv"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

features = []
labels = []

# Loop through each genre folder
for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)
    if os.path.isdir(genre_path):
        print(f"🎵 Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):  # adjust if files are .au
                file_path = os.path.join(genre_path, file)
                mfcc_scaled = extract_features(file_path)
                if mfcc_scaled is not None:
                    features.append(mfcc_scaled)
                    labels.append(genre)

# Save to CSV
if len(features) > 0:
    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Extracted {len(features)} samples and saved to {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted! Check dataset path or file types.")




