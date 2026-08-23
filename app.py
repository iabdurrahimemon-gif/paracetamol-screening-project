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
# NAVIGATION TABS (UPLOAD & ANALYZE ADDED)
# =========================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button(
        "Home",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Home" else "secondary",
    ):
        st.session_state.active_tab = "Home"
        st.rerun()

with col2:
    if st.button(
        "Upload & Analyze",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Upload & Analyze" else "secondary",
    ):
        st.session_state.active_tab = "Upload & Analyze"
        st.rerun()

with col3:
    if st.button(
        "Try the Classifier",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Try the Classifier" else "secondary",
    ):
        st.session_state.active_tab = "Try the Classifier"
        st.rerun()

with col4:
    if st.button(
        "Model Performance",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "Model Performance" else "secondary",
    ):
        st.session_state.active_tab = "Model Performance"
        st.rerun()

with col5:
    if st.button(
        "About the Project",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "About the Project" else "secondary",
    ):
        st.session_state.active_tab = "About the Project"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# TAB 0 — HOME
# =========================================================

if st.session_state.active_tab == "Home":

    st.markdown('<div class="section-title">Rapid Screening Tool for Substandard Paracetamol</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Overview of the framework, operational goals, and design guidelines.</div>',
        unsafe_allow_html=True,
    )

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

    st.markdown(
        """
        <div class="info-card" style="margin-bottom: 1.2rem;">
            <h4>🎯 Intended Use</h4>
            <p>Designed for educational exploration, rapid academic assessment, and preliminary screening simulations to demonstrate how optical data can be paired with machine learning to enhance pharmaceutical quality assurance workflows.</p>
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
            # 1. Upload & Load
            df_raw = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded.")
            
            with st.expander("🔍 Preview Raw Uploaded Data"):
                st.dataframe(df_raw.head(), use_container_width=True)

            # 2. Validate
            required_cols = ["lambda_max", "peak_height", "fwhm", "area_under_curve"]
            missing_cols = [col for col in required_cols if col not in df_raw.columns]

            if missing_cols:
                st.error(f"❌ Validation Error: Missing required columns: `{missing_cols}`. Expected columns: `{required_cols}`")
            else:
                st.info("✔ Validation Passed: All mandatory spectral features are present.")

                # 3. Preprocess
                df_clean = df_raw.dropna(subset=required_cols).copy()
                for col in required_cols:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                df_clean = df_clean.dropna(subset=required_cols)

                st.info(f"✔ Preprocessing Complete: Cleaned dataset contains {len(df_clean)} valid sample rows.")

                # 4. Predict
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

                    # Summary chart of predictions
                    st.markdown("#### Class Distribution Summary")
                    class_counts = pd.Series(predictions).value_counts()
                    
                    fig_batch, ax_batch = plt.subplots(figsize=(7, 2.8))
                    class_counts.plot(kind="bar", ax=ax_batch, color="#2563eb", edgecolor="#1e40af", linewidth=0.6, zorder=3)
                    ax_batch.set_ylabel("Sample Count", labelpad=6)
                    ax_batch.set_xlabel("Classification Category", labelpad=6)
                    plt.xticks(rotation=0)
                    ax_batch.grid(axis="y", alpha=0.6, zorder=0)
                    ax_batch.spines["top"].set_visible(True)
                    ax_batch.spines["right"].set_visible(True)
                    ax_batch.spines["top"].set_color("#94a3b8")
                    ax_batch.spines["right"].set_color("#94a3b8")
                    
                    fig_batch.tight_layout()
                    st.pyplot(fig_batch, use_container_width=True)
                    plt.close(fig_batch)

                    # Download button for processed results
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
# TAB 2 — TRY THE CLASSIFIER
# =========================================================

elif st.session_state.active_tab == "Try the Classifier":

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

    st.markdown(
        f"""
        <div class="standout-confidence">
            <div class="metric-label">Model Confidence Assessment</div>
            <div class="metric-value">{max_probability:.0%} certainty for <em>{display_text}</em></div>
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

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(prob_df["Class"], prob_df["Probability"], color="#2563eb", height=0.5, edgecolor="#1e40af", linewidth=0.6, zorder=3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Model Probability", labelpad=6)
    ax.grid(axis="x", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color("#94a3b8")
    ax.spines["right"].set_color("#94a3b8")
    
    for i, v in enumerate(prob_df["Probability"]):
        ax.text(min(v + 0.03, 0.95), i, f"{v:.1%}", va="center", fontweight="600", color="#1e293b", fontsize=9)
        
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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

    st.markdown("#### Model Explanation (SHAP)")
    st.caption("Detailed breakdown of feature-level impacts on the prediction.")

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_features)

        if isinstance(shap_values, list):
            pred_idx = list(model.classes_).index(prediction)
            shap_val = np.array(shap_values[pred_idx]).flatten()
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            pred_idx = list(model.classes_).index(prediction)
            shap_val = shap_values[0, :, pred_idx]
        else:
            shap_val = np.array(shap_values).flatten()

        shap_df = pd.DataFrame({
            "Feature": list(input_features.columns),
            "SHAP Value": [float(v) for v in shap_val],
            "Feature Value": [float(v) for v in input_features.iloc[0].values]
        }).sort_values("SHAP Value", key=abs, ascending=False)

        fig_shap, ax_shap = plt.subplots(figsize=(7, 2.8))
        colors = ["#059669" if x > 0 else "#dc2626" for x in shap_df["SHAP Value"]]
        ax_shap.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, height=0.5, edgecolor="#64748b", linewidth=0.6, zorder=3)
        ax_shap.set_xlabel("SHAP Value (Impact on Prediction)", labelpad=6)
        ax_shap.axvline(0, color="#64748b", linewidth=1.0, linestyle="-", zorder=2)
        ax_shap.grid(axis="x", alpha=0.6, zorder=0)
        ax_shap.spines["top"].set_visible(True)
        ax_shap.spines["right"].set_visible(True)
        ax_shap.spines["top"].set_color("#94a3b8")
        ax_shap.spines["right"].set_color("#94a3b8")
        
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

        st.info("Positive SHAP values drive the score toward this classification; negative values push it away.")

    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Reconstructed UV-Vis Spectrum</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Gaussian curve representation generated using the input values.</div>',
        unsafe_allow_html=True,
    )

    wavelengths = np.linspace(200, 400, 500)
    sigma = fwhm / 2.355
    simulated_spectrum = peak_height * np.exp(-((wavelengths - lambda_max) ** 2) / (2 * sigma**2))

    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(wavelengths, simulated_spectrum, linewidth=2.4, color="#1d4ed8", zorder=3)
    ax2.fill_between(wavelengths, simulated_spectrum, color="#eff6ff", alpha=0.6, zorder=2)
    ax2.axvline(lambda_max, linestyle="--", linewidth=1.0, color="#d97706", alpha=0.9, zorder=3)
    ax2.scatter([lambda_max], [peak_height], s=55, zorder=4, color="#dc2626", edgecolor="white", linewidth=1.2)
    ax2.annotate(
        f"λmax = {lambda_max:.1f} nm",
        xy=(lambda_max, peak_height),
        xytext=(lambda_max + 12, peak_height * 0.8),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0, alpha=0.8),
        fontsize=9,
        fontweight="600",
        color="#334155"
    )
    ax2.set_xlabel("Wavelength (nm)", labelpad=6)
    ax2.set_ylabel("Absorbance (A.U.)", labelpad=6)
    ax2.set_xlim(200, 400)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.6, zorder=0)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_color("#94a3b8")
    ax2.spines["right"].set_color("#94a3b8")
    
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.info("⚠️ This spectrum plot is a mathematical visualization constructed from parameters, not a raw laboratory output.")


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================

elif st.session_state.active_tab == "Model Performance":

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Evaluation metrics of the Random Forest classifier on the held-out evaluation dataset.</div>',
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
    st.info("The performance data reflects a synthetic dataset built with realistic analytical variance. Treat results as a proof-of-concept.")

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


# =========================================================
# TAB 4 — ABOUT
# =========================================================

elif st.session_state.active_tab == "About the Project":

    st.markdown('<div class="section-title">About the Research Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">From computational molecular modeling to rapid pharmaceutical screening tools.</div>',
        unsafe_allow_html=True,
    )

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
