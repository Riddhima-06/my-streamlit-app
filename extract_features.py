import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def process_dataset(dataset_path="data/genres"):
    features = []
    labels = []

    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue

        for file_name in os.listdir(genre_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(genre_path, file_name)
                mfcc_features = extract_features(file_path)
                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(genre)

    df = pd.DataFrame(features)
    df['label'] = labels
    return df

def main(dataset_path="data/genres", output_file="features.csv"):
    df = process_dataset(dataset_path)
    df.to_csv(output_file, index=False)
    print(f"✅ Features extracted for {len(df)} samples and saved to {output_file}")

if __name__ == "__main__":
    main()


