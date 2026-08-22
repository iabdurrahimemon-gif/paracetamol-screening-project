# Rapid Screening Tool for Substandard/Counterfeit Pharmaceuticals
## UV-Vis Spectral Fingerprinting + Machine Learning + DFT Validation

---

## 1. Problem Statement (write this like a mini research paper)

Substandard and counterfeit medicines are a documented public health and regulatory
problem in Bangladesh and other developing markets. Full confirmatory testing
(HPLC, LC-MS) is accurate but expensive, slow, and requires centralized lab
infrastructure — BSTI/DGDA cannot test every batch in every local market.

UV-Vis spectrophotometry is cheap, fast, and widely available even at smaller
QC labs. This project explores whether a UV-Vis "spectral fingerprint" combined
with a simple ML classifier can flag samples that deviate from a genuine
reference — as a low-cost pre-screening step, not a replacement for confirmatory
testing.

**This framing matters.** You're not claiming to replace HPLC — you're proposing
a triage tool. That's realistic, defensible, and exactly how real regulatory
science thinks about resource-constrained testing.

---

## 2. Data Approach (since you won't have real counterfeit samples)

Be transparent about this in your README — it's normal and expected for a
personal project, and honesty here builds credibility rather than hurting it.

**Step A — Genuine reference spectra**
- Pick 1–2 accessible APIs you can reason about chemically (e.g., Paracetamol —
  well-documented UV-Vis absorbance ~243 nm, widely available reference data
  in pharmacopoeia/literature)
- Either measure it yourself if you still have lab access, OR digitize/reference
  published spectral data, OR generate synthetic spectra using known peak
  position, width, and molar absorptivity (Beer-Lambert + Gaussian peak shape)

**Step B — Simulate "substandard/counterfeit" variants**
Model realistic real-world deviations, not random noise:
- Lower API concentration (common counterfeit issue) → reduced peak height
- Wrong/substitute API → shifted peak position or different peak shape
- Excipient interference/impurities → baseline shift or extra shoulder peaks
- Degraded product → peak broadening or reduced absorbance at expected λmax

**Step C — Add realistic instrument noise**
Small random noise + baseline drift to mimic real spectrophotometer output —
this is what separates a credible simulation from an obviously fake dataset.

---

## 3. Modeling Pipeline

1. **Preprocessing**: baseline correction, normalization of spectra
2. **Feature extraction**: peak position, peak height, peak width (FWHM),
   area under curve — OR use full spectral vector with PCA for dimensionality
   reduction
3. **Classification model**: start simple — Logistic Regression or Random
   Forest on extracted features (genuine vs. suspect). Add PLS-DA
   (Partial Least Squares Discriminant Analysis) if you want the more
   "chemometrics-native" approach real analytical chemists use
4. **Output**: probability/confidence score + flag, not just binary yes/no —
   mirrors how real screening tools report results

---

## 4. DFT Validation Layer (this is your differentiator)

Using Gaussian (which you already have experience with):
- Run a basic DFT calculation (e.g., TD-DFT) on your chosen API molecule to
  predict its theoretical UV-Vis absorption spectrum
- Compare the DFT-predicted λmax against your experimental/simulated genuine
  reference spectrum
- Use this as a **validation/sanity-check step**: "does my reference spectrum
  match what quantum chemistry predicts for this molecule?"

This step alone is rare in a student portfolio — it shows you can move between
computational chemistry, experimental technique, and data science, which is
a very unusual combination for a fresh graduate.

---

## 5. Deliverables (all free tools)

| Component | Tool |
|---|---|
| Data generation & modeling | Python (NumPy, pandas, scikit-learn) in Google Colab |
| Spectral visualization | Matplotlib/Seaborn |
| DFT calculation | Gaussian (you already have access/experience) |
| Interactive demo | Streamlit (upload a spectrum → get a screening result) |
| Hosting | GitHub (code + README) + Streamlit Community Cloud (live demo link) |
| Write-up | README structured like a short paper: Background, Method, Results, Limitations, Future Work |

---

## 6. How to Write the README So It Reads Like Research, Not a Tutorial

Structure:
1. **Motivation** — the real-world problem (cite BSTI/DGDA context, counterfeit
   drug prevalence in the region)
2. **Method** — data simulation approach (be explicit that it's simulated,
   explain why that's a reasonable proxy)
3. **Results** — classification performance, example spectra plots,
   DFT vs experimental comparison
4. **Limitations** — clearly state this is a proof-of-concept, not validated
   against real counterfeit samples, not a regulatory-grade method
5. **Future Work** — how this could be extended (more APIs, real sample
   validation, partnership with a lab)

Section 4 (Limitations) is what separates a credible project from an
overclaiming one — interviewers respect honesty about scope far more than
inflated claims.

---

## 7. Suggested Timeline

- **Week 1**: Literature/background research on paracetamol UV-Vis behavior,
  set up Colab, generate genuine + simulated substandard spectra
- **Week 2**: Build preprocessing + feature extraction + classifier, evaluate
  performance
- **Week 3**: Run DFT calculation in Gaussian, compare to experimental/simulated
  spectrum
- **Week 4**: Build Streamlit demo, write README, polish GitHub repo, deploy

---

## 8. One-Line Pitch (for CV/interview)

*"Built a low-cost UV-Vis + machine learning screening tool to flag potentially
substandard pharmaceutical samples, validated the reference spectrum against
DFT-predicted absorption using Gaussian — a proof-of-concept for accessible
pharmaceutical quality screening in resource-limited settings."*
