"""
Phase 2: Feature Extraction
Project: Rapid Screening Tool for Substandard Pharmaceuticals

Takes the raw spectra from Phase 1 (500 wavelength points per sample) and
extracts a small set of meaningful features that a classifier can learn from:

- lambda_max      : wavelength of peak absorbance (nm)
- peak_height      : absorbance value at lambda_max
- fwhm             : Full Width at Half Maximum (nm) - standard spectroscopy
                     measure of band width
- area_under_curve : total absorbance across the scan range (proxy for
                     total concentration / trapezoidal integration)

These four features turn a 500-column raw spectrum into a compact,
interpretable feature vector - exactly what a real analytical chemist
would look at when comparing spectra by hand, just automated.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Load Phase 1 dataset
# ---------------------------------------------------------
df = pd.read_csv("paracetamol_synthetic_spectra.csv")

# Wavelength columns are everything except the "label" column
wavelength_cols = [c for c in df.columns if c != "label"]
wavelengths = np.array([float(c.replace("nm", "")) for c in wavelength_cols])

# ---------------------------------------------------------
# 2. Feature extraction functions
# ---------------------------------------------------------
def get_lambda_max_and_height(spectrum, wavelengths):
    """Finds the wavelength and absorbance value at the peak."""
    peak_idx = np.argmax(spectrum)
    return wavelengths[peak_idx], spectrum[peak_idx]

def get_fwhm(spectrum, wavelengths, peak_idx):
    """
    Full Width at Half Maximum: the width of the peak measured at
    half its maximum height. Standard spectroscopy metric for band width.
    """
    half_max = spectrum[peak_idx] / 2

    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
        right_idx += 1

    return wavelengths[right_idx] - wavelengths[left_idx]

def get_area_under_curve(spectrum, wavelengths):
    """Trapezoidal integration of the spectrum - proxy for total signal."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(spectrum, wavelengths)
    return np.trapz(spectrum, wavelengths)

# ---------------------------------------------------------
# 3. Extract features for every sample
# ---------------------------------------------------------
features = []

for _, row in df.iterrows():
    spectrum = row[wavelength_cols].values.astype(float)
    peak_idx = np.argmax(spectrum)

    lambda_max, peak_height = get_lambda_max_and_height(spectrum, wavelengths)
    fwhm = get_fwhm(spectrum, wavelengths, peak_idx)
    area = get_area_under_curve(spectrum, wavelengths)

    features.append({
        "lambda_max": lambda_max,
        "peak_height": peak_height,
        "fwhm": fwhm,
        "area_under_curve": area,
        "label": row["label"],
    })

feature_df = pd.DataFrame(features)

# ---------------------------------------------------------
# 4. Save and summarize
# ---------------------------------------------------------
feature_df.to_csv("paracetamol_extracted_features.csv", index=False)

print("Feature extraction complete.")
print(f"Dataset shape: {feature_df.shape}\n")
print("Feature averages by class:")
print(feature_df.groupby("label")[["lambda_max", "peak_height", "fwhm", "area_under_curve"]].mean().round(2))