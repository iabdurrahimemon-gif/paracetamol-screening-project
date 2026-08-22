"""
Phase 3: Classification Model
Project: Rapid Screening Tool for Substandard Pharmaceuticals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("paracetamol_extracted_features.csv")

feature_cols = ["lambda_max", "peak_height", "fwhm", "area_under_curve"]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix: Paracetamol Sample Classification")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.savefig("phase3_confusion_matrix.png", dpi=150)
print("\nSaved confusion matrix as phase3_confusion_matrix.png")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances.round(3))

plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="steelblue")
plt.xlabel("Importance")
plt.title("Which Spectral Feature Matters Most for Classification?")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("phase3_feature_importance.png", dpi=150)
print("Saved feature importance plot as phase3_feature_importance.png")

joblib.dump(model, "paracetamol_classifier_model.joblib")
print("\nModel saved as paracetamol_classifier_model.joblib")