import pandas as pd
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Dataset
# Make sure this matches your actual CSV filename in the Datasets folder
dataset_path = "Datasets/Training.csv" 
# Note: If you are using 'symptoms_df.csv' for training, change the path above.
# Usually, these projects have a 'Training.csv' with 1s and 0s and a 'prognosis' column.

try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Could not find {dataset_path}")
    exit()

# 2. Prepare Data (X = Symptoms, y = Disease)
# We drop the 'prognosis' column for X, and use it for y
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# 3. Train/Test Split (Optional, but good for checking accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# IMPORTANT: probability=True is required for calculation_confidence() in main.py
print("⏳ Training model... (This might take a few seconds)")
svc = SVC(kernel='linear', probability=True) 
svc.fit(X_train, y_train)

# 5. Check Accuracy
predictions = svc.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model
import os
os.makedirs('models', exist_ok=True)
with open('models/svc.pkl', 'wb') as f:
    pickle.dump(svc, f)
print("💾 Model saved to models/svc.pkl")

# ==========================================================
# 7. IMPORTANT: GENERATE ID MAPPING FOR MAIN.PY
# ==========================================================
# Your main.py uses a dictionary 'diseases_list = {0: ..., 1: ...}'
# We need to see which number the model assigned to which disease.
print("\n⚠️ COPY THIS DICTIONARY IF IT IS DIFFERENT FROM YOUR MAIN.PY ⚠️")
print("="*60)
disease_mapping = {i: label for i, label in enumerate(svc.classes_)}
print(disease_mapping)
print("="*60)