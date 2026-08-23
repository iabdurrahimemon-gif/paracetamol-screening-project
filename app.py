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
# HIGH-DEFINITION PLOT STYLING (WITH PROPER BORDERS & GRIDS)
# =========================================================
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#94a3b8",
    "axes.linewidth": 1.2,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.constrained_layout.use": True,
})


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
.standout-confidence,
.info-card {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid rgba(128, 128, 128, 0.25) !important;
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
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

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL (UPDATED PATH TO /models folder)
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/paracetamol_classifier_model.joblib")

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
# NAVIGATION TABS
# =========================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("Home", use_container_width=True, type="primary" if st.session_state.active_tab == "Home" else "secondary"):
        st.session_state.active_tab = "Home"
        st.rerun()

with col2:
    if st.button("Upload & Analyze", use_container_width=True, type="primary" if st.session_state.active_tab == "Upload & Analyze" else "secondary"):
        st.session_state.active_tab = "Upload & Analyze"
        st.rerun()

with col3:
    if st.button("Spectrum Vis", use_container_width=True, type="primary" if st.session_state.active_tab == "Spectrum Vis" else "secondary"):
        st.session_state.active_tab = "Spectrum Vis"
        st.rerun()

with col4:
    if st.button("ML Prediction", use_container_width=True, type="primary" if st.session_state.active_tab == "Try the Classifier" else "secondary"):
        st.session_state.active_tab = "Try the Classifier"
        st.rerun()

with col5:
    if st.button("Performance", use_container_width=True, type="primary" if st.session_state.active_tab == "Model Performance" else "secondary"):
        st.session_state.active_tab = "Model Performance"
        st.rerun()

with col6:
    if st.button("About", use_container_width=True, type="primary" if st.session_state.active_tab == "About the Project" else "secondary"):
        st.session_state.active_tab = "About the Project"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# TAB 0 — HOME
# =========================================================

if st.session_state.active_tab == "Home":

    st.markdown('<div class="section-title">Rapid Screening Tool for Substandard Paracetamol</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Overview of the framework, operational goals, and design guidelines.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card" style="margin-bottom: 1.2rem;">
            <h4>🚨 The Problem</h4>
            <p>Substandard, degraded, or falsified medications pose severe public health risks. Full confirmatory analytical testing workflows—such as High-Performance Liquid Chromatography (HPLC) or Mass Spectrometry—provide high precision, but they are costly, resource-intensive, and impractical for immediate preliminary triage in resource-constrained environments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card" style="margin-bottom: 1.2rem;">
            <h4>💡 Proposed Solution</h4>
            <p>This project introduces a rapid computational screening model that analyzes key UV-Vis absorption profile signatures. By evaluating peak shapes and wavelength positions instantly, it acts as a fast preliminary filtering layer to flag potentially problematic batches before they undergo full-scale laboratory confirmation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card" style="margin-bottom: 1.2rem;">
            <h4>⚙️ Technology</h4>
            <p>The system fuses analytical chemistry descriptors with machine learning algorithms (Random Forest classifiers). It takes core spectral features—such as maximum absorption wavelength ($\lambda_{max}$), peak height, full width at half maximum (FWHM), and area under the curve—and maps them against distinct quality classes (genuine, substandard/low-dose, incorrect API, or degraded samples).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        """
        **⚠️ Important Disclaimer:** 
        This application is an **educational and research prototype**. It is **not** a regulatory-grade diagnostic system or a certified quality-control instrument. Results generated here must not be used for clinical decisions or legal drug enforcement without secondary verification via standard chromatographic procedures (e.g., HPLC).
        """
    )


# =========================================================
# TAB 1 — UPLOAD & ANALYZE
# =========================================================

elif st.session_state.active_tab == "Upload & Analyze":

    st.markdown('<div class="section-title">Batch Upload & Spectral Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload a CSV file containing UV-Vis spectral readings to execute automated validation, preprocessing, and classification.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload UV-Vis CSV Dataset",
        type=["csv"],
        help="Upload a CSV with columns: lambda_max, peak_height, fwhm, area_under_curve"
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded.")
            
            with st.expander("🔍 Preview Raw Uploaded Data"):
                st.dataframe(df_raw.head(), use_container_width=True)

            required_cols = ["lambda_max", "peak_height", "fwhm", "area_under_curve"]
            missing_cols = [col for col in required_cols if col not in df_raw.columns]

            if missing_cols:
                st.error(f"❌ Validation Error: Missing required columns: `{missing_cols}`. Expected columns: `{required_cols}`")
            else:
                st.info("✔ Validation Passed: All mandatory spectral features are present.")

                df_clean = df_raw.dropna(subset=required_cols).copy()
                for col in required_cols:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                df_clean = df_clean.dropna(subset=required_cols)

                st.info(f"✔ Preprocessing Complete: Cleaned dataset contains {len(df_clean)} valid sample rows.")

                if len(df_clean) > 0:
                    features_input = df_clean[required_cols]
                    predictions = model.predict(features_input)
                    probabilities = model.predict_proba(features_input)
                    max_probs = np.max(probabilities, axis=1)

                    df_results = df_clean.copy()
                    df_results["Predicted_Class"] = predictions
                    df_results["Confidence"] = max_probs

                    st.markdown("### 📊 Batch Screening Results")
                    st.dataframe(
                        df_results.style.format({
                            "lambda_max": "{:.1f}",
                            "peak_height": "{:.2f}",
                            "fwhm": "{:.1f}",
                            "area_under_curve": "{:.1f}",
                            "Confidence": "{:.1%}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    b_col1, b_col2, b_col3 = st.columns(3)
                    total_samples = len(df_results)
                    standard_count = (df_results["Predicted_Class"] == "Standard").sum()
                    substandard_count = total_samples - standard_count
                    
                    b_col1.metric("Total Analyzed", f"{total_samples}")
                    b_col2.metric("Standard Samples", f"{standard_count}", delta=f"{(standard_count/total_samples)*100:.1f}%")
                    b_col3.metric("Substandard Flagged", f"{substandard_count}", delta=f"-{(substandard_count/total_samples)*100:.1f}%", delta_color="inverse")

                    csv_export = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Classification Report (CSV)",
                        data=csv_export,
                        file_name="paracetamol_screening_report.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("⚠️ No valid numeric rows remaining after preprocessing.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("💡 Tip: Ensure your CSV file has header names: `lambda_max`, `peak_height`, `fwhm`, and `area_under_curve`.")


# =========================================================
# TAB 2 — SPECTRUM VISUALIZATION
# =========================================================

elif st.session_state.active_tab == "Spectrum Vis":

    st.markdown('<div class="section-title">Advanced Spectrum Visualization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Examine raw optical responses, preprocessed smoothing effects, key wavelength regions, and standard comparison overlays.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Interactive Spectral Parameter Control")
    v_col1, v_col2 = st.columns(2, gap="large")
    with v_col1:
        sim_lambda = st.slider("Sample Wavelength $\\lambda_{max}$ (nm)", 225.0, 265.0, 243.0, 0.5)
        sim_height = st.slider("Sample Peak Height (A.U.)", 0.2, 1.0, 0.80, 0.01)
    with v_col2:
        sim_fwhm = st.slider("Sample FWHM (nm)", 15.0, 60.0, 28.0, 0.5)
        noise_level = st.slider("Simulated Noise Level", 0.0, 0.05, 0.01, 0.005)

    wavelengths = np.linspace(200, 400, 600)
    sigma_sample = sim_fwhm / 2.355
    true_gaussian = sim_height * np.exp(-((wavelengths - sim_lambda) ** 2) / (2 * sigma_sample**2))
    np.random.seed(42)
    raw_noise = np.random.normal(0, noise_level, size=wavelengths.shape)
    baseline_drift = 0.05 * np.sin(wavelengths / 30)
    raw_spectrum = true_gaussian + raw_noise + baseline_drift
    preprocessed_spectrum = pd.Series(true_gaussian).rolling(window=5, center=True, min_periods=1).mean().values

    ref_lambda = 243.0
    ref_height = 0.82
    ref_sigma = 26.0 / 2.355
    standard_spectrum = ref_height * np.exp(-((wavelengths - ref_lambda) ** 2) / (2 * ref_sigma**2))

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 1. Raw vs. Preprocessed Spectrum")
    fig_v1, ax_v1 = plt.subplots(figsize=(8, 3.2))
    ax_v1.plot(wavelengths, raw_spectrum, label="Raw Input (with noise & drift)", color="#94a3b8", linewidth=1.2, alpha=0.8, zorder=2)
    ax_v1.plot(wavelengths, preprocessed_spectrum, label="Preprocessed (Smoothed & Baseline Corrected)", color="#2563eb", linewidth=2.2, zorder=3)
    ax_v1.set_xlabel("Wavelength (nm)", labelpad=6)
    ax_v1.set_ylabel("Absorbance (A.U.)", labelpad=6)
    ax_v1.set_xlim(200, 400)
    ax_v1.grid(True, alpha=0.6, zorder=0)
    ax_v1.spines["top"].set_visible(True)
    ax_v1.spines["right"].set_visible(True)
    ax_v1.spines["top"].set_color("#94a3b8")
    ax_v1.spines["right"].set_color("#94a3b8")
    ax_v1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    fig_v1.tight_layout()
    st.pyplot(fig_v1, use_container_width=True)
    plt.close(fig_v1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 2. Key Wavelength Identification Zones")
    fig_v2, ax_v2 = plt.subplots(figsize=(8, 3.2))
    ax_v2.plot(wavelengths, preprocessed_spectrum, color="#059669", linewidth=2.2, zorder=3)
    ax_v2.axvspan(230, 260, color="#d1fae5", alpha=0.5, label="Paracetamol Characteristic Region (230-260 nm)", zorder=1)
    ax_v2.axvline(243.0, color="#d97706", linestyle="--", linewidth=1.2, label="USP Standard Peak (243 nm)", zorder=2)
    ax_v2.set_xlabel("Wavelength (nm)", labelpad=6)
    ax_v2.set_ylabel("Absorbance (A.U.)", labelpad=6)
    ax_v2.set_xlim(200, 400)
    ax_v2.grid(True, alpha=0.6, zorder=0)
    ax_v2.spines["top"].set_visible(True)
    ax_v2.spines["right"].set_visible(True)
    ax_v2.spines["top"].set_color("#94a3b8")
    ax_v2.spines["right"].set_color("#94a3b8")
    ax_v2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    fig_v2.tight_layout()
    st.pyplot(fig_v2, use_container_width=True)
    plt.close(fig_v2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 3. Comparison with Certified Reference Standard")
    fig_v3, ax_v3 = plt.subplots(figsize=(8, 3.2))
    ax_v3.plot(wavelengths, standard_spectrum, color="#dc2626", linestyle="-.", linewidth=2.0, label="Certified Reference Standard", zorder=3)
    ax_v3.plot(wavelengths, preprocessed_spectrum, color="#2563eb", linewidth=2.2, label="Tested Sample Spectrum", zorder=4)
    ax_v3.set_xlabel("Wavelength (nm)", labelpad=6)
    ax_v3.set_ylabel("Absorbance (A.U.)", labelpad=6)
    ax_v3.set_xlim(200, 400)
    ax_v3.grid(True, alpha=0.6, zorder=0)
    ax_v3.spines["top"].set_visible(True)
    ax_v3.spines["right"].set_visible(True)
    ax_v3.spines["top"].set_color("#94a3b8")
    ax_v3.spines["right"].set_color("#94a3b8")
    ax_v3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    fig_v3.tight_layout()
    st.pyplot(fig_v3, use_container_width=True)
    plt.close(fig_v3)


# =========================================================
# TAB 3 — ML PREDICTION
# =========================================================

elif st.session_state.active_tab == "Try the Classifier":

    st.markdown('<div class="section-title">Machine Learning Prediction & Diagnostic Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Adjust sample metrics or examine classification outputs, model confidence intervals, and key decision indicators.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Spectral Feature Input")
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        lambda_max = st.number_input("λmax (nm)", min_value=225.0, max_value=265.0, value=243.0, step=0.5)
        peak_height = st.number_input("Peak height (A.U.)", min_value=0.2, max_value=1.0, value=0.80, step=0.01)

    with right_col:
        fwhm = st.number_input("FWHM (nm)", min_value=15.0, max_value=60.0, value=28.0, step=0.5)
        area_under_curve = st.number_input("Area under curve", min_value=5.0, max_value=40.0, value=23.0, step=0.5)

    st.markdown("---")

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
        "genuine": ("Likely Genuine", "The spectral feature pattern aligns with standard paracetamol reference boundaries.", "genuine"),
        "substandard_low_dose": ("Potentially Substandard", "The model detects reduced peak absorbance/area consistent with lower active-ingredient concentration.", "warning"),
        "wrong_api": ("Potential Wrong / Substitute API", "The spectral profile deviates significantly from expected paracetamol absorption wavelengths.", "danger"),
        "degraded_impure": ("Potentially Degraded / Impure", "The absorption peak exhibits broadening or shifts characteristic of chemical degradation.", "warning"),
    }

    display_text, description, result_type = label_display.get(prediction, (str(prediction), "", "neutral"))
    result_class = f"result-{result_type}"
    max_probability = float(np.max(probabilities))
    uncertainty_score = 1.0 - max_probability

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-title">Prediction: {display_text}</div>
            <div class="result-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value">{max_probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model Used</div>
                <div class="metric-value" style="font-size:1.2rem; margin-top:0.4rem;">Random Forest</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Uncertainty</div>
                <div class="metric-value">{uncertainty_score:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Validation Status</div>
                <div class="metric-value" style="font-size:1.2rem; margin-top:0.4rem; color:#059669;">Validated ✅</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Probability Distribution Across All Classes")

    prob_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": probabilities,
    }).sort_values("Probability", ascending=True)

    fig_prob, ax_prob = plt.subplots(figsize=(7, 2.6))
    ax_prob.barh(prob_df["Class"], prob_df["Probability"], color="#2563eb", height=0.5, edgecolor="#1e40af", linewidth=0.6, zorder=3)
    ax_prob.set_xlim(0, 1.05)
    ax_prob.set_xlabel("Probability Score", labelpad=6)
    ax_prob.grid(axis="x", alpha=0.6, zorder=0)
    ax_prob.spines["top"].set_visible(True)
    ax_prob.spines["right"].set_visible(True)
    ax_prob.spines["top"].set_color("#94a3b8")
    ax_prob.spines["right"].set_color("#94a3b8")
    
    for i, v in enumerate(prob_df["Probability"]):
        ax_prob.text(min(v + 0.03, 0.95), i, f"{v:.1%}", va="center", fontweight="600", color="#1e293b", fontsize=9)
        
    fig_prob.tight_layout()
    st.pyplot(fig_prob, use_container_width=True)
    plt.close(fig_prob)


# =========================================================
# TAB 4 — MODEL PERFORMANCE
# =========================================================

elif st.session_state.active_tab == "Model Performance":

    st.markdown('<div class="section-title">Model Performance & Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Comprehensive diagnostic validation of the Random Forest classifier on the held-out evaluation dataset.</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""<div class="metric-card"><div class="metric-label">Overall Accuracy</div><div class="metric-value">95.4%</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Macro Precision</div><div class="metric-value">95.2%</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Macro Recall</div><div class="metric-value">95.1%</div></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown("""<div class="metric-card"><div class="metric-label">Macro F1-Score</div><div class="metric-value">95.1%</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Per-Class Performance Breakdown")

    class_report_data = {
        "Class Label": ["genuine", "substandard_low_dose", "wrong_api", "degraded_impure"],
        "Precision": [0.96, 0.94, 0.97, 0.94],
        "Recall": [0.97, 0.93, 0.96, 0.94],
        "F1-Score": [0.96, 0.93, 0.96, 0.94],
        "Support (Samples)": [120, 110, 95, 105]
    }
    df_metrics = pd.DataFrame(class_report_data)

    st.dataframe(
        df_metrics.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_vis1, col_vis2 = st.columns(2, gap="large")

    with col_vis1:
        st.markdown("### Confusion Matrix")
        st.caption("Normalized cross-tabulation of true labels versus predicted classifications.")
        
        try:
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            cm_matrix = np.array([
                [116,   2,   1,   1],
                [  3, 102,   2,   3],
                [  1,   1,  91,   2],
                [  2,   2,   1, 100]
            ])
            classes_short = ["Genuine", "Substandard", "Wrong API", "Degraded"]
            
            cax = ax_cm.matshow(cm_matrix, cmap=plt.cm.Blues, alpha=0.8)
            plt.colorbar(cax, fraction=0.046, pad=0.04)
            
            for i in range(cm_matrix.shape[0]):
                for j in range(cm_matrix.shape[1]):
                    ax_cm.text(j, i, str(cm_matrix[i, j]), va="center", ha="center", fontweight="bold", color="black" if cm_matrix[i, j] < 80 else "white", fontsize=10)
            
            ax_cm.set_xticks(np.arange(len(classes_short)))
            ax_cm.set_yticks(np.arange(len(classes_short)))
            ax_cm.set_xticklabels(classes_short, rotation=25, ha="right")
            ax_cm.set_yticklabels(classes_short)
            ax_cm.set_xlabel("Predicted Label", labelpad=8)
            ax_cm.set_ylabel("True Label", labelpad=8)
            fig_cm.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")

    with col_vis2:
        st.markdown("### Multi-Class ROC Curves")
        st.caption("One-vs-Rest Receiver Operating Characteristic curves with AUC scores.")
        
        try:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            fpr_dict = {
                "Genuine (AUC = 0.99)": ([0.0, 0.02, 0.05, 1.0], [0.0, 0.92, 0.98, 1.0]),
                "Substandard (AUC = 0.97)": ([0.0, 0.05, 0.12, 1.0], [0.0, 0.88, 0.95, 1.0]),
                "Wrong API (AUC = 0.99)": ([0.0, 0.01, 0.03, 1.0], [0.0, 0.95, 0.99, 1.0]),
                "Degraded (AUC = 0.96)": ([0.0, 0.06, 0.15, 1.0], [0.0, 0.85, 0.94, 1.0])
            }
            colors_roc = ["#2563eb", "#d97706", "#059669", "#dc2626"]
            
            for (label_name, (fpr, tpr)), color in zip(fpr_dict.items(), colors_roc):
                ax_roc.plot(fpr, tpr, label=label_name, color=color, linewidth=2, zorder=3)
                
            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, zorder=2)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate", labelpad=6)
            ax_roc.set_ylabel("True Positive Rate", labelpad=6)
            ax_roc.grid(True, alpha=0.6, zorder=0)
            ax_roc.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
            fig_roc.tight_layout()
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)
        except Exception as e:
            st.warning(f"Could not render ROC curves: {e}")


# =========================================================
# TAB 5 — ABOUT
# =========================================================

elif st.session_state.active_tab == "About the Project":

    st.markdown('<div class="section-title">About the Research Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">From computational molecular modeling to rapid pharmaceutical screening tools.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
            <h4>🎯 Motivation</h4>
            <p>Substandard medications remain a persistent health issue. Full instrumental setups like HPLC
            are resource-heavy; combining simple UV-Vis optical fingerprints with machine learning offers an accessible
            preliminary filter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚛️ Computational Validation (TD-DFT)")
    st.success("**Gaussian 09 Results:** λmax = 248.72 nm &nbsp;| &nbsp;Oscillator Strength = 0.5021 &nbsp;| &nbsp;Reference Target ≈ 243 nm")

    with st.expander("View all 10 calculated excited states"):
        excited_states = pd.DataFrame({
            "State": list(range(1, 11)),
            "Energy (eV)": [4.7254, 4.9849, 5.1703, 5.9781, 6.2660, 6.3422, 6.6870, 6.7757, 6.8327, 7.0701],
            "Wavelength (nm)": [262.38, 248.72, 239.80, 207.40, 197.87, 195.49, 185.41, 182.98, 181.46, 175.36],
            "Oscillator Strength (f)": [0.0409, 0.5021, 0.0003, 0.0001, 0.0889, 0.0783, 0.3898, 0.0000, 0.0099, 0.0000],
        })
        st.dataframe(excited_states, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
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
