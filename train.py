# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


def main():
    # Load features
    if not os.path.exists("features.csv"):
        print("Error: features.csv not found.")
        return

    df = pd.read_csv("features.csv")

    if "label" not in df.columns:
        print("Error: 'label' column missing in features.csv")
        return

    X = df.drop("label", axis=1)
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save model and scaler
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    main()
