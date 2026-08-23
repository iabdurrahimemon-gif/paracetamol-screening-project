"""
Phase 5: Rigorous Model Validation & Comparison Pipeline
Project: Rapid Screening Tool for Substandard Pharmaceuticals
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Classifiers to compare
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# =========================================================
# 1. DATA GENERATION / LOADING & ENCODING
# =========================================================
def generate_synthetic_spectral_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # Classes: 0: genuine, 1: substandard_low_dose, 2: wrong_api, 3: degraded_impure
    classes = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.3, 0.15, 0.2])
    
    data = []
    for c in classes:
        if c == 0:  # Genuine: lambda_max ~ 243nm, sharp peak, narrow FWHM
            l_max = np.random.normal(243.0, 0.8)
            height = np.random.normal(0.82, 0.04)
            fwhm = np.random.normal(26.0, 1.5)
            auc = np.random.normal(22.5, 1.2)
            label = "genuine"
        elif c == 1:  # Substandard / Low Dose: lower peak height and AUC
            l_max = np.random.normal(243.2, 1.2)
            height = np.random.normal(0.45, 0.06)
            fwhm = np.random.normal(27.0, 2.0)
            auc = np.random.normal(12.0, 1.5)
            label = "substandard_low_dose"
        elif c == 2:  # Wrong API: significant lambda_max shift
            l_max = np.random.normal(275.0, 3.5)
            height = np.random.normal(0.75, 0.08)
            fwhm = np.random.normal(35.0, 3.0)
            auc = np.random.normal(28.0, 2.5)
            label = "wrong_api"
        else:  # Degraded / Impure: broadened FWHM and distorted peak
            l_max = np.random.normal(245.5, 2.0)
            height = np.random.normal(0.60, 0.07)
            fwhm = np.random.normal(48.0, 4.0)
            auc = np.random.normal(25.0, 2.0)
            label = "degraded_impure"
            
        data.append([l_max, height, fwhm, auc, label])
        
    df = pd.DataFrame(data, columns=["lambda_max", "peak_height", "fwhm", "area_under_curve", "target"])
    return df

print("Generating/Loading dataset for multi-model evaluation...")
df = generate_synthetic_spectral_dataset(n_samples=1500)

# Map text labels to numeric integers for XGBoost and sklearn compatibility
label_mapping = {
    "genuine": 0,
    "substandard_low_dose": 1,
    "wrong_api": 2,
    "degraded_impure": 3
}
df["target_encoded"] = df["target"].map(label_mapping)

X = df[["lambda_max", "peak_height", "fwhm", "area_under_curve"]]
y = df["target_encoded"]

# Inverse mapping for readable reports later
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# =========================================================
# 2. TRAIN / TEST SPLIT (Stratified to maintain class ratios)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)} | Independent Test samples: {len(X_test)}")

# =========================================================
# 3. MODEL COMPARISON USING CROSS-VALIDATION
# =========================================================
models = {
    "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    "Support Vector Machine": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Cross-Validation Performance (5-Fold Stratified Accuracy) ---")
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name:25s} | Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Identify best model based on CV score
best_model_name = max(cv_results, key=cv_results.get)
print(f"\n🏆 Best Performing Model via Cross-Validation: {best_model_name}")

# =========================================================
# 4. FINAL TRAINING & INDEPENDENT TEST EVALUATION
# =========================================================
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Evaluate on unseen independent test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Independent Test Set Evaluation ({best_model_name}) ---")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

# Check for overfitting by comparing train vs test accuracy
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
if abs(train_accuracy - test_accuracy) < 0.05:
    print("✅ Model generalization is healthy (No severe overfitting detected).")
else:
    print("⚠️ Potential overfitting detected between training and test sets.")

# =========================================================
# 5. ARTIFACT EXPORT
# =========================================================
os.makedirs("models", exist_ok=True)
model_path = "models/paracetamol_classifier_model.joblib"
joblib.dump(best_model, model_path)
print(f"\n💾 Successfully saved optimized model to '{model_path}'")