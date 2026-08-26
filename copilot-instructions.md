# PakSat Air Quality Analyzer & Clinical Triage System — Copilot Instructions

## Project Context
PakSat is an Environmental Health Intelligence (EHI) platform operating across Pakistan (Lahore, Karachi, Islamabad)[cite: 2, 5]. It fuses satellite observations (NASA MODIS AOD_047, Sentinel-5P NO2), ERA5 meteorological reanalysis, and OpenAQ ground sensors to estimate ground-level PM2.5 using LightGBM regressors[cite: 2, 5]. Out-of-sample target accuracy: R² ≈ 0.945, RMSE ≈ 9.91 µg/m³[cite: 2, 5]. Model outputs feed directly into an AI Healthcare Triage System for clinical risk escalation and emergency department surge forecasting[cite: 2, 5].

---

## Tech Stack & Primary Tools
- **Language**: Python 3.10+
- **Data & Remote Sensing**: `earthengine-api` (Google Earth Engine), `pandas`, `numpy`, `scikit-learn`
- **Machine Learning Engine**: `lightgbm` (LightGBMRegressor)
- **Web Interface & GIS**: `streamlit`, `folium`, `streamlit-folium`
- **GCP Target**: Project `alibaba-hackathon-505817`[cite: 2, 5]
- **Document Pipeline**: `python-docx`, `weasyprint`

---

## Key Domain Physics & Feature Engineering Rules
When assisting with feature engineering or model logic, strictly enforce these domain formulations:

1. **3D Planetary Boundary Layer Height (PBLH) Inversion**:
   - Satellites capture 2D column optical depth (AOD)[cite: 2, 5]. Vertical compression MUST be calculated using PBLH and wind speed[cite: 2, 5].
   - **Atmospheric Stagnation Index**: `1000 / (wind_speed * pblh + 1)`[cite: 2, 5]
   - **Thermal Confinement Ratio**: `temperature / (pblh / 100)`[cite: 2, 5]

2. **Photochemical Secondary Inorganic Aerosol (SIA) Proxy**:
   - Capture non-linear smog formation using: `photochemical_pm25_proxy = NO2_density * temperature * AOD_047`[cite: 2, 5].
   - Apply **Hygroscopic Growth Correction**: `1 / (1 - relative_humidity)` to strip water-weight particle distortion[cite: 2, 5].

3. **Feature Importance Hierarchy**:
   - Primary features: `AOD_047` (~51.8% weight), `NO2_density` (~25.4% weight), `photochemical_pm25_proxy` (~18.8% weight)[cite: 2, 5].

---

## Coding Standards & Guidelines

### Python & Machine Learning
- Prefer vectorized `pandas` / `numpy` operations over loops for data processing.
- Ensure all model training scripts maintain spatial/temporal cross-validation splits across test cities (Lahore, Karachi, Islamabad) to prevent data leakage[cite: 2, 5].
- Always handle missing satellite/meteorological data gracefully (e.g., cloud masking or iterative interpolation) before passing inputs to LightGBM.

### Streamlit & Visualization (`app.py`)
- Use `@st.cache_data` or `@st.cache_resource` for loading `lightgbm_pm25_model.pkl` and heavy geospatial layers.
- Map layers must be rendered cleanly using `folium` with color-coded risk bands for PM2.5 levels.

### Healthcare Triage System
- If estimated PM2.5 > 150 µg/m³, escalate clinical severity to `Environmental Acute Exacerbation`[cite: 2, 5].
- Recommend evidence-based interventions: N95 respirator mandates, HEPA indoor filtration, and bronchodilator precautions[cite: 2, 5].

---

## What to Avoid (Anti-Patterns)
- **NO** hardcoded GCP credentials or GEE auth keys; use environment variables or native GEE authentication.
- **NO** linear ML models for PM2.5 prediction without non-linear interaction terms.
- **NO** fabricated accuracy claims—keep predictions aligned with empirical data metrics ($R^2 \approx 0.9450$, $\text{RMSE} \approx 9.91\,\mu\text{g}/\text{m}^3$)[cite: 2, 5].
