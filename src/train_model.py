import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =============================
# SETTINGS
# =============================

DATA_PATH = "data/raw/mudra_landmarks.csv"
MODEL_PATH = "models/mudra_classifier.pkl"


# =============================
# LANDMARK NORMALIZATION
# =============================

def normalize_landmarks(row):
    """
    Normalize 21 hand landmarks.

    Landmark 0 (wrist) becomes the origin.
    Coordinates are then scaled based on hand size.
    """

    landmarks = row.values.astype(float).reshape(21, 3)

    # Wrist becomes origin
    landmarks = landmarks - landmarks[0]

    # Calculate hand size
    distances = np.linalg.norm(landmarks, axis=1)
    scale = np.max(distances)

    # Avoid division by zero
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# =============================
# LOAD DATASET
# =============================

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df)} samples")
print(f"Columns: {len(df.columns)}")


# =============================
# SEPARATE FEATURES / LABEL
# =============================

X_raw = df.drop(columns=["label"])
y = df["label"]

print("\nClasses:")
print(y.value_counts())


# =============================
# NORMALIZE FEATURES
# =============================

print("\nNormalizing landmarks...")

X = np.array([
    normalize_landmarks(row)
    for _, row in X_raw.iterrows()
])

print(f"Normalized feature shape: {X.shape}")


# =============================
# TRAIN / TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# =============================
# CREATE CLASSIFIER
# =============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# =============================
# TRAIN MODEL
# =============================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# =============================
# EVALUATE MODEL
# =============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =============================
# SAVE MODEL
# =============================

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")