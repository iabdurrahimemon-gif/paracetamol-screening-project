"""
Phase 1 (v2): Synthetic UV-Vis Spectra Generator - REALISTIC OVERLAP VERSION
Project: Rapid Screening Tool for Substandard Pharmaceuticals
Compound: Paracetamol (Acetaminophen)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wavelengths = np.linspace(200, 400, 500)

def gaussian_peak(x, center, height, width):
    return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

GENUINE_LAMBDA_MAX = 243
GENUINE_HEIGHT = 0.80
GENUINE_WIDTH = 12

def add_instrument_noise(spectrum, noise_level=0.025, baseline_drift=0.035):
    noise = np.random.normal(0, noise_level, size=spectrum.shape)
    drift = baseline_drift * np.linspace(0, 1, len(spectrum)) * np.random.choice([-1, 1])
    return spectrum + noise + drift

def generate_genuine():
    factor = np.random.uniform(0.85, 1.15)
    lambda_jitter = np.random.uniform(-4, 4)
    width_jitter = np.random.uniform(-3, 3)
    spectrum = gaussian_peak(
        wavelengths,
        GENUINE_LAMBDA_MAX + lambda_jitter,
        GENUINE_HEIGHT * factor,
        GENUINE_WIDTH + width_jitter,
    )
    return add_instrument_noise(spectrum)

def generate_substandard_low_dose():
    factor = np.random.uniform(0.55, 0.90)
    lambda_jitter = np.random.uniform(-4, 4)
    spectrum = gaussian_peak(
        wavelengths, GENUINE_LAMBDA_MAX + lambda_jitter, GENUINE_HEIGHT * factor, GENUINE_WIDTH
    )
    return add_instrument_noise(spectrum)

def generate_wrong_api():
    shift = np.random.uniform(3, 18) * np.random.choice([-1, 1])
    spectrum = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX + shift, GENUINE_HEIGHT, GENUINE_WIDTH)
    return add_instrument_noise(spectrum)

def generate_degraded_impure():
    width_increase = np.random.uniform(1.05, 1.8)
    main = gaussian_peak(wavelengths, GENUINE_LAMBDA_MAX, GENUINE_HEIGHT * np.random.uniform(0.75, 0.95), GENUINE_WIDTH * width_increase)
    spectrum = main
    if np.random.rand() < 0.7:
        shoulder_center = GENUINE_LAMBDA_MAX + np.random.uniform(12, 25)
        shoulder = gaussian_peak(wavelengths, shoulder_center, GENUINE_HEIGHT * np.random.uniform(0.1, 0.25), 8)
        spectrum = spectrum + shoulder
    return add_instrument_noise(spectrum)

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

if __name__ == "__main__":
    dataset = build_dataset(n_per_class=60)
    dataset.to_csv("paracetamol_synthetic_spectra.csv", index=False)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset["label"].value_counts())

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
    plt.title("Simulated UV-Vis Spectra: Genuine vs Substandard Paracetamol Variants (v2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase1_example_spectra.png", dpi=150)
    print("Saved plot as phase1_example_spectra.png")