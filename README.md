\# Rapid Screening Tool for Substandard Pharmaceuticals

\### UV-Vis Spectral Fingerprinting + Machine Learning + DFT Validation



\## Motivation



Substandard and counterfeit medicines are a documented public health and

regulatory challenge in Bangladesh and other developing markets. Full

confirmatory testing (HPLC, LC-MS) is accurate but expensive, slow, and

requires centralized lab infrastructure — regulatory bodies like BSTI/DGDA

cannot test every batch in every local market.



UV-Vis spectrophotometry is cheap, fast, and widely available even at

smaller QC labs. This project explores whether a UV-Vis "spectral

fingerprint," combined with a simple machine learning classifier, can flag

samples that deviate from a genuine reference — as a low-cost pre-screening

step, not a replacement for confirmatory testing.



\*\*This is a proof-of-concept personal project, not a validated diagnostic

tool.\*\* See Limitations below.



\## Project Status



🚧 \*\*In progress\*\* — Phase 1 complete, Phase 2 underway.



| Phase | Description | Status |

|---|---|---|

| 1 | Synthetic spectra generation | ✅ Complete |

| 2 | Preprocessing + feature extraction | 🔄 In progress |

| 3 | ML classifier (genuine vs. substandard/wrong API/degraded) | ⬜ Planned |

| 4 | DFT validation of reference spectrum (Gaussian) | ✅ Complete |

| 5 | Interactive Streamlit demo | ⬜ Planned |



\## Method



\### 1. Data Simulation



Since real counterfeit pharmaceutical samples aren't accessible for a

personal project, spectra are simulated based on known chemical behavior of

paracetamol (λmax ≈ 243 nm in aqueous solution) and realistic failure modes:



\- \*\*Genuine\*\* — correct λmax, expected peak height (normal concentration)

\- \*\*Substandard (low dose)\*\* — same λmax, reduced peak height (lower

&#x20; concentration)

\- \*\*Wrong/substitute API\*\* — shifted λmax, simulating a different

&#x20; active ingredient

\- \*\*Degraded/impure\*\* — broadened peak with a secondary shoulder,

&#x20; simulating a degradation product



Each spectrum also includes randomized instrument noise and baseline drift

to mimic real spectrophotometer output.



\### 2. Feature Extraction \*(Phase 2, in progress)\*



Peak position, peak height, peak width (FWHM), and area under the curve

will be extracted from each spectrum as inputs to the classifier.



\### 3. Classification \*(Phase 3, planned)\*



A classifier (starting with Logistic Regression / Random Forest, potentially

extended to PLS-DA) will be trained to flag genuine vs. suspect samples.



\### 4. DFT Validation



The genuine reference spectrum's λmax was independently cross-checked

against a TD-DFT prediction using Gaussian 09 (B3LYP/6-31G(d,p), PCM

water solvent model, 10 excited states).



\*\*Result:\*\* the dominant electronic transition (by oscillator strength)

was predicted at \*\*λmax = 248.72 nm (f = 0.5021)\*\*, compared to the

experimental/reference value of \*\*\~243 nm\*\* — a difference of \*\*5.72 nm\*\*.



This level of agreement is consistent with the well-documented systematic

overestimation of vertical excitation energies by B3LYP for π→π\*

transitions, and supports the validity of the reference spectrum used

to generate the synthetic dataset in Phase 1.



\*Note: an initial attempt used the larger 6-311+G(d,p) basis set, but

this repeatedly crashed the available 32-bit Gaussian 09 installation

(IA32W build) due to memory addressing limits. Switching to 6-31G(d,p) —

still a solid double-zeta polarized basis set — resolved this while

still giving chemically meaningful results.\*



\## Repository Structure



```

paracetamol-screening-project/

├── files/                              # working files/backups

├── paracetamol\_synthetic\_spectra.csv   # Phase 1 dataset (240 samples)

├── phase1\_generate\_spectra.py          # Phase 1 script

├── phase1\_example\_spectra.png          # sanity-check plot

├── project-plan-spectral-screening.md  # full project plan

└── README.md

```



\## Tools Used



Python (NumPy, pandas, matplotlib, scikit-learn), Gaussian (DFT), Streamlit,

Git/GitHub — all free tools.



\## Limitations



\- Spectra are \*\*simulated\*\*, not measured from real samples — this is a

&#x20; reasonable proxy for demonstrating method and pipeline, not a substitute

&#x20; for lab-validated data

\- This is a \*\*proof-of-concept screening aid\*\*, not a regulatory-grade or

&#x20; clinically validated diagnostic method

\- Currently limited to a single API (paracetamol); extending to other

&#x20; compounds would require new reference data and re-validation



\## Future Work



\- Extend to additional active pharmaceutical ingredients

\- Validate against real instrument data where accessible

\- Explore partnership with a lab for real-sample testing



\---



\*Author: Abdur Rahim Emon — M.S. Inorganic Chemistry, University of

Chittagong\*



