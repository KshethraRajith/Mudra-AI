import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = "data/raw/mudra_landmarks.csv"
MODEL_PATH = "models/mudra_classifier.pkl"

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df)} samples")
print(f"Columns: {len(df.columns)}")


# -----------------------------
# 2. Separate features and labels
# -----------------------------
X = df.drop(columns=["label"])
y = df["label"]

print("\nClasses:")
print(y.value_counts())


# -----------------------------
# 3. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -----------------------------
# 4. Create classifier
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# 5. Train model
# -----------------------------
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------
# 6. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 7. Save model
# -----------------------------
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")