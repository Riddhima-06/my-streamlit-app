import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # Load extracted features
    df = pd.read_csv("features.csv")

    # X = features, y = labels (make sure your dataset has a "label" column with genre names)
    X = df.drop(columns=["label"])
    y = df["label"]   # e.g., "rock", "classical", etc.

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model (Random Forest for example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained with accuracy: {acc:.2f}")

    # Save model & scaler in current working directory
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    print("✅ Model and scaler saved in:", os.getcwd())

if __name__ == "__main__":
    main()
