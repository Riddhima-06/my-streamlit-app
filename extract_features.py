import os
import librosa
import numpy as np
import pandas as pd

# Function to extract features from one audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # load audio
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)

        # Combine features into a single vector
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
        return features
    except Exception as e:
        print(f"❌ Error extracting {file_path}: {e}")
        return None

# Main function to extract from dataset
def main():
    dataset_path = "data/genres"   # your dataset folder
    data = []
    labels = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                genre = os.path.basename(root)  # folder name = genre
                features = extract_features(file_path)

                if features is not None:
                    data.append(features)
                    labels.append(genre)

    # Save to DataFrame
    if data:
        df = pd.DataFrame(data)
        df["label"] = labels
        df.to_csv("features.csv", index=False)
        print("✅ Features extracted and saved to features.csv")
    else:
        print("⚠️ No features extracted. Check dataset path.")

if __name__ == "__main__":
    main()




