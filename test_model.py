import pickle
import numpy as np
import pandas as pd

# ======================================================
# HealixAI - Model Test Script
# ======================================================

# -----------------------------
# Load trained model
# -----------------------------
with open("models/healix_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

# -----------------------------
# Load training columns (feature names)
# -----------------------------
df = pd.read_csv("Datasets/Training.csv")
symptom_columns = df.columns[:-1].tolist()

print("✅ Symptom columns loaded")

# -----------------------------
# Example user symptoms (test input)
# -----------------------------
user_symptoms = ["itching", "skin_rash", "fatigue"]

# -----------------------------
# Create input vector
# -----------------------------
input_vector = np.zeros(len(symptom_columns))

for symptom in user_symptoms:
    symptom = symptom.strip().lower()
    if symptom in symptom_columns:
        idx = symptom_columns.index(symptom)
        input_vector[idx] = 1

# Convert to DataFrame (IMPORTANT)
X_input = pd.DataFrame([input_vector], columns=symptom_columns)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(X_input)[0]

confidence = None
if hasattr(model, "predict_proba"):
    confidence = model.predict_proba(X_input).max() * 100

# -----------------------------
# Output
# -----------------------------
print("\n🩺 HealixAI Test Result")
print("-----------------------")
print("Predicted Disease:", prediction)

if confidence is not None:
    print("Confidence:", round(confidence, 2), "%")
else:
    print("Confidence: Not available")
