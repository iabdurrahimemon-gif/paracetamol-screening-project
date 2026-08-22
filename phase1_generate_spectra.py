"""
Phase 1: Synthetic UV-Vis Spectra Generator
Project: Rapid Screening Tool for Substandard Pharmaceuticals
Compound: Paracetamol (Acetaminophen)

This script generates realistic synthetic UV-Vis spectra representing:
- Genuine samples (correct concentration, correct API)
- Substandard samples (low concentration)
- Wrong/substitute API samples (shifted peak)
- Degraded/impure samples (broadened or shouldered peak)

All spectra include realistic instrument noise and baseline drift.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define wavelength range (typical UV-Vis scan range)
# ---------------------------------------------------------
wavelengths = np.linspace(200, 400, 500)  # 200-400 nm, 500 points

# ---------------------------------------------------------
# 2. Core function: Gaussian peak (approximates absorbance band)
# ---------------------------------------------------------
def gaussian_peak(x, center, height, width):
    """Simulates a single absorbance band using a Gaussian shape."""
    return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

# ---------------------------------------------------------
# 3. Reference parameters for genuine Paracetamol
# ---------------------------------------------------------
GENUINE_LAMBDA_MAX = 243   # nm, literature value for paracetamol
GENUINE_HEIGHT = 0.80      # arbitrary absorbance units at 100% expected conc.
GENUINE_WIDTH = 12         # nm, typical band width

def add_instrument_noise(spectrum, noise_level=0.01, baseline_drift=0.02):
    """Adds realistic random noise + slight baseline drift."""
    noise = np.random.normal(0, noise_level, size=spectrum.shape)
    drift = baseline_drift * np.linspace(0, 1, len(spectrum)) * np.random.choice([-1, 1])
    return spectrum + noise + drift

# ---------------------------------------------------------
# 4. Sample generators for each class
# ---------------------------------------------------------
def generate_genuine():
    spectrum = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX, GENUINE_HEIGHT, GENUINE_WIDTH)
    # small natural batch-to-batch variation (genuine samples aren't identical)
    spectrum *= np.random.uniform(0.95, 1.05)
    return add_instrument_noise(spectrum)

def generate_substandard_low_dose():
    # 40-70% of expected concentration -> reduced peak height, same lambda_max
    factor = np.random.uniform(0.4, 0.7)
    spectrum = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX, GENUINE_HEIGHT * factor, GENUINE_WIDTH)
    return add_instrument_noise(spectrum)

def generate_wrong_api():
    # shifted lambda_max, simulating a different/substitute compound
    shift = np.random.uniform(8, 20) * np.random.choice([-1, 1])
    spectrum = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX + shift, GENUINE_HEIGHT, GENUINE_WIDTH)
    return add_instrument_noise(spectrum)

def generate_degraded_impure():
    # broadened main peak + small shoulder peak from a degradation product
    width_increase = np.random.uniform(1.3, 1.8)
    main = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX, GENUINE_HEIGHT * 0.85, GENUINE_WIDTH * width_increase)
    shoulder_center = GENUINE_LAMBDA_MAX + np.random.uniform(15, 25)
    shoulder = gaussian_peak(wavelengths, shoulder_center, GENUINE_HEIGHT * 0.25, 8)
    spectrum = main + shoulder
    return add_instrument_noise(spectrum)

# ---------------------------------------------------------
# 5. Build the dataset
# ---------------------------------------------------------
def build_dataset(n_per_class=60, seed=42):
    np.random.seed(seed)
    rows = []
    labels = []
    generators = {
        "genuine": generate_genuine,
        "substandard_low_dose": generate_substandard_low_dose,
        "wrong_api": generate_wrong_api,
        "degraded_impure": generate_degraded_impure,
    }
    for label, func in generators.items():
        for _ in range(n_per_class):
            rows.append(func())
            labels.append(label)

    df = pd.DataFrame(rows, columns=[f"{wl:.1f}nm" for wl in wavelengths])
    df["label"] = labels
    return df

# ---------------------------------------------------------
# 6. Generate and save
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset = build_dataset(n_per_class=60)
    dataset.to_csv("paracetamol_synthetic_spectra.csv", index=False)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset["label"].value_counts())

    # Plot one example of each class for a sanity check
    plt.figure(figsize=(9, 6))
    examples = {
        "Genuine": generate_genuine(),
        "Substandard (low dose)": generate_substandard_low_dose(),
        "Wrong API (shifted peak)": generate_wrong_api(),
        "Degraded/Impure": generate_degraded_impure(),
    }
    for label, spectrum in examples.items():
        plt.plot(wavelengths, spectrum, label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (A.U.)")
    plt.title("Simulated UV-Vis Spectra: Genuine vs Substandard Paracetamol Variants")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase1_example_spectra.png", dpi=150)
    print("Saved plot as phase1_example_spectra.png")
