"""
Phase 5: Interactive Streamlit Demo
Project: Rapid Screening Tool for Substandard Pharmaceuticals
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Paracetamol Screening | UV-Vis + ML",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
 <style>

/* ---------- GLOBAL ---------- */
.stApp {
    background-color: var(--background-color);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* ---------- HEADER ---------- */
.hero {
    background: linear-gradient(135deg, #0b2545 0%, #123f68 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(11, 37, 69, 0.25);
}

.hero-title {
    color: white !important;
    font-size: 2.5rem;
    font-weight: 750;
    margin: 0;
    letter-spacing: -0.8px;
}

.hero-subtitle {
    color: #dbeafe !important;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #e0f2fe !important;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ---------- SECTION TITLES ---------- */
.section-title {
    color: var(--text-color) !important;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    color: var(--text-color) !important;
    opacity: 0.75;
    font-size: 0.92rem;
    margin-bottom: 1.4rem;
}

/* ---------- CARDS ---------- */
.metric-card,
.standout-class,
.standout-confidence,
.info-card {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid rgba(128, 128, 128, 0.25) !important;
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
}

.standout-class {
    border: 2px solid #10b981 !important;
}

.standout-confidence {
    border: 2px solid #3b82f6 !important;
}

.metric-label {
    color: var(--text-color) !important;
    opacity: 0.7;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-value {
    color: var(--text-color) !important;
    font-size: 1.65rem;
    font-weight: 750;
    margin-top: 0.25rem;
}

.metric-unit {
    color: var(--text-color) !important;
    opacity: 0.7;
    font-size: 0.8rem;
}

/* ---------- RESULT BOXES ---------- */
.result-genuine {
    background-color: rgba(16, 185, 129, 0.12) !important;
    border-left: 6px solid #10b981 !important;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

.result-warning {
    background-color: rgba(245, 158, 11, 0.12) !important;
    border-left: 6px solid #f59e0b !important;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

.result-danger {
    background-color: rgba(239, 68, 68, 0.12) !important;
    border-left: 6px solid #ef4444 !important;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

.result-neutral {
    background-color: rgba(59, 130, 246, 0.12) !important;
    border-left: 6px solid #3b82f6 !important;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

.result-title {
    color: var(--text-color) !important;
    font-size: 1.45rem;
    font-weight: 750;
}

.result-description {
    color: var(--text-color) !important;
    opacity: 0.8;
    font-size: 0.9rem;
    margin-top: 0.35rem;
}

/* ---------- CUSTOM BUTTON TABS ---------- */
div.stButton > button {
    border-radius: 12px !important;
    height: 46px !important;
    font-weight: 600 !important;
    border: 2px solid #94a3b8 !important;
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"] {
    background-color: #0b4f82 !important;
    border-color: #0b4f82 !important;
    color: white !important;
    font-weight: 700 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(11, 79, 130, 0.3) !important;
}

/* ---------- SLIDERS ---------- */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span {
    color: var(--text-color) !important;
    font-weight: 700 !important;
}

/* ---------- OTHER ---------- */
.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.6;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(128,128,128,0.2);
}

.pipeline-step {
    background-color: rgba(11, 79, 130, 0.08) !important;
    border-left: 5px solid #0b4f82 !important;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}

.pipeline-step strong {
    color: var(--text-color) !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("paracetamol_classifier_model.joblib")

model = load_model()


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Rapid Pharmaceutical Screening</div>
        <div class="hero-subtitle">
            UV-Vis Spectral Fingerprinting + Machine Learning
            for Paracetamol Quality Screening
        </div>
        <div class="hero-badge">
            RESEARCH PROTOTYPE · PARACETAMOL PROOF-OF-CONCEPT
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CUSTOM BUTTON TABS
# =========================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Try the Classifier"

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(
        "Try the Classifier",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Try the Classifier" else "secondary",
    ):
        st.session_state.active_tab = "Try the Classifier"
        st.rerun()

with col2:
    if st.button(
        "Model Performance",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Model Performance" else "secondary",
    ):
        st.session_state.active_tab = "Model Performance"
        st.rerun()

with col3:
    if st.button(
        "About the Project",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "About the Project" else "secondary",
    ):
        st.session_state.active_tab = "About the Project"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# TAB 1 — CLASSIFIER
# =========================================================

if st.session_state.active_tab == "Try the Classifier":

    st.markdown(
        '<div class="section-title">Interactive Spectral Screening</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Adjust the spectral features below and evaluate how the trained classifier interprets the sample.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Spectral Input")
    st.caption("Type exact values on the left **or** use the sliders on the right.")

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("**Exact Values**")
        lambda_max = st.number_input(
            "λmax (nm)", min_value=225.0, max_value=265.0, value=243.0, step=0.5,
            help="Expected paracetamol absorption maximum ≈ 243 nm"
        )
        peak_height = st.number_input(
            "Peak height (A.U.)", min_value=0.2, max_value=1.0, value=0.80, step=0.01
        )
        fwhm = st.number_input(
            "FWHM (nm)", min_value=15.0, max_value=60.0, value=28.0, step=0.5
        )
        area_under_curve = st.number_input(
            "Area under curve", min_value=5.0, max_value=40.0, value=23.0, step=0.5
        )

    with right_col:
        st.markdown("**Sliders**")
        lambda_max = st.slider(
            "λmax — Peak wavelength", min_value=225.0, max_value=265.0,
            value=float(lambda_max), step=0.5
        )
        peak_height = st.slider(
            "Peak height — Absorbance", min_value=0.2, max_value=1.0,
            value=float(peak_height), step=0.01
        )
        fwhm = st.slider(
            "FWHM — Peak width", min_value=15.0, max_value=60.0,
            value=float(fwhm), step=0.5
        )
        area_under_curve = st.slider(
            "Area under curve", min_value=5.0, max_value=40.0,
            value=float(area_under_curve), step=0.5
        )

    st.markdown("---")

    # Prediction
    input_features = pd.DataFrame([{
        "lambda_max": lambda_max,
        "peak_height": peak_height,
        "fwhm": fwhm,
        "area_under_curve": area_under_curve,
    }])

    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    class_labels = model.classes_

    label_display = {
        "genuine": (
            "Likely Genuine",
            "The spectral feature pattern is consistent with the genuine reference class.",
            "genuine",
        ),
        "substandard_low_dose": (
            "Potentially Substandard",
            "The model detects a spectral pattern consistent with reduced active-ingredient concentration.",
            "warning",
        ),
        "wrong_api": (
            "Potential Wrong / Substitute API",
            "The predicted spectral pattern differs from the expected paracetamol reference.",
            "danger",
        ),
        "degraded_impure": (
            "Potentially Degraded / Impure",
            "The spectral characteristics are consistent with altered or broadened absorption.",
            "warning",
        ),
    }

    display_text, description, result_type = label_display.get(
        prediction, (str(prediction), "", "neutral")
    )
    result_class = f"result-{result_type}"
    max_probability = float(np.max(probabilities))

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-title">{display_text}</div>
            <div class="result-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
            <div class="standout-class">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value">{display_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="standout-confidence">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{max_probability:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": probabilities,
    }).sort_values("Probability", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(prob_df["Class"], prob_df["Probability"], color="#1e40af")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model probability")
    ax.grid(axis="x", alpha=0.2)
    for i, v in enumerate(prob_df["Probability"]):
        ax.text(min(v + 0.02, 0.94), i, f"{v:.0%}", va="center")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Feature Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Feature Summary</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    metrics = [
        ("λmax", f"{lambda_max:.1f}", "nm"),
        ("Peak Height", f"{peak_height:.2f}", "A.U."),
        ("FWHM", f"{fwhm:.1f}", "nm"),
        ("Area", f"{area_under_curve:.1f}", "A.U.·nm"),
    ]
    for col, (label, value, unit) in zip([k1, k2, k3, k4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
# =====================================================
# SHAP EXPLANATION
# =====================================================
st.markdown("#### Model Explanation (SHAP)")
st.caption("This shows how each spectral feature contributed to the current prediction.")

try:
    import shap

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_features)

    # For multi-class, we take the SHAP values of the predicted class
    if isinstance(shap_values, list):
        # Find index of predicted class
        pred_idx = list(model.classes_).index(prediction)
        shap_val = shap_values[pred_idx][0]
    else:
        shap_val = shap_values[0]

    # Create a nice dataframe for display
    shap_df = pd.DataFrame({
        "Feature": input_features.columns,
        "SHAP Value": shap_val,
        "Feature Value": input_features.iloc[0].values
    }).sort_values("SHAP Value", key=abs, ascending=False)

    # Bar chart of SHAP values
    fig_shap, ax_shap = plt.subplots(figsize=(8, 3.5))
    colors = ["#10b981" if x > 0 else "#ef4444" for x in shap_df["SHAP Value"]]
    ax_shap.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax_shap.set_xlabel("SHAP Value (Impact on Prediction)")
    ax_shap.axvline(0, color="gray", linewidth=0.8)
    ax_shap.grid(axis="x", alpha=0.3)
    fig_shap.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close(fig_shap)

    st.dataframe(
        shap_df.style.format({
            "SHAP Value": "{:.4f}",
            "Feature Value": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info("Positive SHAP values push the prediction toward the predicted class. Negative values push it away.")

except Exception as e:
    st.warning(f"SHAP explanation could not be generated: {e}")

    # Spectrum
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Reconstructed UV-Vis Spectrum</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A Gaussian-shaped absorption band reconstructed from the four input features.</div>',
        unsafe_allow_html=True,
    )

    wavelengths = np.linspace(200, 400, 500)
    sigma = fwhm / 2.355
    simulated_spectrum = peak_height * np.exp(-((wavelengths - lambda_max) ** 2) / (2 * sigma**2))

    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    ax2.plot(wavelengths, simulated_spectrum, linewidth=2.5, color="#1e40af")
    ax2.axvline(lambda_max, linestyle="--", linewidth=1, alpha=0.6, color="#f59e0b")
    ax2.scatter([lambda_max], [peak_height], s=55, zorder=5, color="#ef4444")
    ax2.annotate(
        f"λmax = {lambda_max:.1f} nm",
        xy=(lambda_max, peak_height),
        xytext=(lambda_max + 10, peak_height * 0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.5),
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Absorbance (A.U.)")
    ax2.set_xlim(200, 400)
    ax2.grid(alpha=0.18)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.info("⚠️ The spectrum shown above is reconstructed from the selected features for visualization. It is not a raw experimental spectrum.")


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

elif st.session_state.active_tab == "Model Performance":

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Evaluation of the Random Forest classifier on the held-out test set.</div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""<div class="metric-card"><div class="metric-label">Test Accuracy</div><div class="metric-value">95%</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value" style="font-size:1.35rem;">Random Forest</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Input Features</div><div class="metric-value">4</div></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown("""<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">4</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("The dataset is **synthetic** and was designed with realistic class overlap and instrument-like variability. Therefore, the reported performance should be interpreted as **proof-of-concept** rather than real-world clinical or regulatory performance.")

    perf_col1, perf_col2 = st.columns(2, gap="large")
    with perf_col1:
        st.markdown("### Confusion Matrix")
        try:
            st.image("phase3_confusion_matrix.png", use_container_width=True)
        except Exception:
            st.warning("Confusion matrix image not found.")
    with perf_col2:
        st.markdown("### Feature Importance")
        try:
            st.image("phase3_feature_importance.png", use_container_width=True)
        except Exception:
            st.warning("Feature importance image not found.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>Interpretation</h4>
            <p>The classifier achieved <strong>95% test accuracy</strong> on the synthetic held-out dataset.
            The most meaningful potential confusion occurs between <strong>genuine</strong> and
            <strong>low-dose</strong> samples, which is chemically plausible because their spectral
            characteristics can partially overlap.</p>
            <p style="margin-bottom:0;">Feature importance analysis shows that <strong>λmax</strong> and <strong>peak height</strong>
            are the most discriminative descriptors for separating the four quality classes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TAB 3 — ABOUT
# =========================================================

elif st.session_state.active_tab == "About the Project":

    st.markdown('<div class="section-title">About the Research Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">From simulated UV-Vis spectra to machine-learning-based pharmaceutical screening.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🎯 Motivation</h4>
            <p>Substandard and counterfeit medicines represent an important quality-control challenge.
            Full confirmatory analytical methods such as HPLC and LC-MS provide high confidence
            but may not be practical for rapid screening of every sample.</p>
            <p style="margin-bottom:0;">This project explores whether a low-cost UV-Vis spectral fingerprint, combined with
            machine learning, could provide a preliminary screening step before confirmatory
            laboratory analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 Analytical Pipeline")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(
            """
            <div class="pipeline-step"><strong>01 — Synthetic Spectra Generation</strong><br>
            Simulated UV-Vis spectra for four classes:<br>
            • Genuine paracetamol<br>• Substandard / low-dose<br>
            • Wrong or substitute API<br>• Degraded / impure material</div>
            <div class="pipeline-step"><strong>02 — Feature Extraction</strong><br>
            Four spectral descriptors extracted:<br>
            • λmax &nbsp;• Peak height &nbsp;• FWHM &nbsp;• Area under the curve</div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="pipeline-step"><strong>03 — Machine Learning</strong><br>
            A Random Forest classifier was trained using the extracted spectral features.</div>
            <div class="pipeline-step"><strong>04 — Computational Validation</strong><br>
            The reference absorption region was independently compared with a TD-DFT calculation using Gaussian 09.</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ⚛️ DFT Validation")
    st.success("**TD-DFT prediction:** λmax = 248.72 nm  |  Oscillator strength = 0.5021  |  Experimental/reference ≈ 243 nm  |  Difference = 5.72 nm")
    st.write("The dominant electronic transition among the ten calculated excited states was predicted at 248.72 nm with an oscillator strength of 0.5021.")

    with st.expander("View all 10 calculated excited states"):
        excited_states = pd.DataFrame({
            "State": list(range(1, 11)),
            "Energy (eV)": [4.7254, 4.9849, 5.1703, 5.9781, 6.2660, 6.3422, 6.6870, 6.7757, 6.8327, 7.0701],
            "Wavelength (nm)": [262.38, 248.72, 239.80, 207.40, 197.87, 195.49, 185.41, 182.98, 181.46, 175.36],
            "Oscillator Strength (f)": [0.0409, 0.5021, 0.0003, 0.0001, 0.0889, 0.0783, 0.3898, 0.0000, 0.0099, 0.0000],
        })
        st.dataframe(excited_states, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Limitations")
    st.warning(
        """
        **Important scientific limitations**
        - The spectra used are **simulated**, not measured from real samples.
        - This is a **proof-of-concept** and has not been validated on real pharmaceutical batches.
        - Not a regulatory-grade quality-control method.
        - Confirmatory testing (e.g. HPLC) is still required.
        - Currently limited to **paracetamol**.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h4>👨‍🔬 Researcher</h4>
            <p style="margin-bottom:0.4rem;"><strong>Abdur Rahim Emon</strong><br>
            M.S. Inorganic Chemistry<br>University of Chittagong</p>
            <p style="margin-bottom:0; color:#64748b;">Project: Rapid Screening Tool for Substandard Pharmaceuticals</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Rapid Pharmaceutical Screening Tool · UV-Vis + Machine Learning<br>
        Research Prototype · Not a Substitute for Confirmatory Pharmaceutical QC
    </div>
    """,
    unsafe_allow_html=True,
)
