"""
Phase 5: Interactive Streamlit Demo
Project: Rapid Screening Tool for Substandard Pharmaceuticals

Run with: streamlit run app.py

Three sections:
1. Try the Classifier  - manually input spectral features, get a live prediction
2. Model Performance    - confusion matrix + feature importance from Phase 3
3. About the Project    - methodology write-up, including a DFT validation
                          placeholder to fill in once the Gaussian results are in
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Paracetamol Screening Tool",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------
# Load trained model (Phase 3 output)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("paracetamol_classifier_model.joblib")

model = load_model()

# ---------------------------------------------------------
# Load reference data for realistic slider ranges
# ---------------------------------------------------------
@st.cache_data
def load_features():
    return pd.read_csv("paracetamol_extracted_features.csv")

features_df = load_features()

st.title("🧪 Rapid Screening Tool for Substandard Pharmaceuticals")
st.caption("UV-Vis Spectral Fingerprinting + Machine Learning | Paracetamol proof-of-concept")

tab1, tab2, tab3 = st.tabs(["Try the Classifier", "Model Performance", "About the Project"])

# ===========================================================
# TAB 1: Interactive Classifier
# ===========================================================
with tab1:
    st.subheader("Enter spectral features to get a live prediction")
    st.write(
        "Adjust the sliders below to simulate a UV-Vis spectral reading, "
        "or enter your own measured values. The model will predict whether "
        "the sample looks genuine or shows signs of being substandard."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        lambda_max = st.slider(
            "λmax — Peak wavelength (nm)",
            min_value=225.0, max_value=265.0, value=243.0, step=0.5,
            help="Wavelength of maximum absorbance. Genuine paracetamol peaks near 243 nm.",
        )
        peak_height = st.slider(
            "Peak Height (Absorbance, A.U.)",
            min_value=0.2, max_value=1.0, value=0.80, step=0.01,
            help="Higher = more concentrated. Substandard (low-dose) samples show reduced height.",
        )
        fwhm = st.slider(
            "FWHM — Peak Width (nm)",
            min_value=15.0, max_value=60.0, value=28.0, step=0.5,
            help="Width of the absorbance band. Degraded/impure samples show broader peaks.",
        )
        area_under_curve = st.slider(
            "Area Under Curve",
            min_value=5.0, max_value=40.0, value=23.0, step=0.5,
            help="Total absorbance signal across the scan range.",
        )

    input_features = pd.DataFrame([{
        "lambda_max": lambda_max,
        "peak_height": peak_height,
        "fwhm": fwhm,
        "area_under_curve": area_under_curve,
    }])

    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    class_labels = model.classes_

    with col2:
        st.markdown("### Prediction")

        label_display = {
            "genuine": ("✅ Genuine", "This sample's spectral signature matches expected genuine paracetamol."),
            "substandard_low_dose": ("⚠️ Substandard (Low Dose)", "Reduced concentration signal detected."),
            "wrong_api": ("🚫 Wrong/Substitute API", "Peak position doesn't match expected paracetamol wavelength."),
            "degraded_impure": ("⚠️ Degraded / Impure", "Broadened peak shape suggests degradation or impurities."),
        }

        display_text, description = label_display.get(prediction, (prediction, ""))
        st.markdown(f"## {display_text}")
        st.write(description)

        st.markdown("#### Confidence by class")
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(prob_df["Class"], prob_df["Probability"], color="steelblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.02, i, f"{v:.0%}", va="center")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### What this spectrum looks like")

    wavelengths = np.linspace(200, 400, 500)
    simulated_spectrum = peak_height * np.exp(-((wavelengths - lambda_max) ** 2) / (2 * (fwhm / 2.355) ** 2))

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(wavelengths, simulated_spectrum, color="steelblue", linewidth=2)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Absorbance (A.U.)")
    ax2.set_title("Reconstructed Spectrum from Your Inputs")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ===========================================================
# TAB 2: Model Performance
# ===========================================================
with tab2:
    st.subheader("How well does the classifier actually perform?")
    st.write(
        "These results are on a held-out test set (samples the model never saw during training), "
        "using a synthetic dataset designed with realistic, overlapping class boundaries — "
        "not artificially perfect separation."
    )

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**Confusion Matrix**")
        try:
            st.image("phase3_confusion_matrix.png", use_container_width=True)
        except Exception:
            st.info("Run phase3_train_classifier.py first to generate this image.")

    with perf_col2:
        st.markdown("**Feature Importance**")
        try:
            st.image("phase3_feature_importance.png", use_container_width=True)
        except Exception:
            st.info("Run phase3_train_classifier.py first to generate this image.")

    st.markdown("---")
    st.markdown("**Test Accuracy: 95%** — the only meaningful confusion is between "
                "*genuine* and *substandard (low-dose)* samples, which makes chemical "
                "sense since their concentration ranges were designed to overlap slightly, "
                "mirroring real borderline QC cases.")

# ===========================================================
# TAB 3: About the Project
# ===========================================================
with tab3:
    st.subheader("Methodology")

    st.markdown("""
**Motivation:** Substandard and counterfeit medicines are a documented regulatory
challenge in markets like Bangladesh, where full confirmatory testing (HPLC, LC-MS)
isn't feasible for every batch. This project explores whether a cheap, fast UV-Vis
"spectral fingerprint," combined with machine learning, can serve as a low-cost
pre-screening step — not a replacement for confirmatory testing.

**Pipeline:**
1. **Synthetic spectra generation** — realistic UV-Vis spectra simulated for genuine,
   substandard (low-dose), wrong/substitute API, and degraded/impure paracetamol samples,
   with deliberate class overlap and instrument noise to mirror real analytical data.
2. **Feature extraction** — λmax, peak height, FWHM, and area under the curve pulled
   from each raw spectrum.
3. **Classification** — a Random Forest classifier trained on these features, achieving
   95% test accuracy with chemically sensible confusion patterns.
4. **DFT validation** — the genuine reference spectrum's λmax was independently
   cross-checked against a TD-DFT prediction (Gaussian 09, B3LYP/6-31G(d,p), PCM water
   solvent model, 10 excited states).
""")

    st.markdown("### DFT Validation Results")
    st.success(
        "✅ **TD-DFT prediction complete.**\n\n"
        "The dominant electronic transition (highest oscillator strength among 10 "
        "computed excited states) was predicted at **λmax = 248.72 nm** (f = 0.5021), "
        "compared to the experimental/reference value of **~243 nm** — a difference "
        "of **5.72 nm**.\n\n"
        "This level of agreement is consistent with the well-documented systematic "
        "overestimation of vertical excitation energies by B3LYP for π→π* transitions, "
        "and supports the validity of the reference spectrum used throughout this project."
    )

    with st.expander("See all 10 computed excited states"):
        excited_states = pd.DataFrame({
            "State": list(range(1, 11)),
            "Energy (eV)": [4.7254, 4.9849, 5.1703, 5.9781, 6.2660, 6.3422, 6.6870, 6.7757, 6.8327, 7.0701],
            "Wavelength (nm)": [262.38, 248.72, 239.80, 207.40, 197.87, 195.49, 185.41, 182.98, 181.46, 175.36],
            "Oscillator Strength (f)": [0.0409, 0.5021, 0.0003, 0.0001, 0.0889, 0.0783, 0.3898, 0.0000, 0.0099, 0.0000],
        })
        st.dataframe(excited_states, use_container_width=True, hide_index=True)
        st.caption("State 2 (bold in discussion above) has by far the highest oscillator "
                   "strength and corresponds to the experimentally observable absorption band.")

    st.markdown("""
### Limitations
- Spectra are **simulated**, not measured from real samples — a reasonable proxy for
  demonstrating the pipeline, not a substitute for lab-validated data
- This is a **proof-of-concept screening aid**, not a regulatory-grade diagnostic method
- Currently limited to paracetamol; extending to other APIs would need new reference data

### Author
Abdur Rahim Emon — M.S. Inorganic Chemistry, University of Chittagong
""")
