# Zameen Environmental Health Intelligence — Copilot Instructions

## Project Context
Zameen is an Environmental Health Intelligence decision-support prototype for hospitals and government. Its current pilot covers Lahore, Karachi, and Islamabad. It can combine available ground observations with satellite and meteorological features for PM2.5 estimation. It does not currently provide a validated hospital-demand forecast or nationally representative pollution estimates. Never claim model performance until it has been measured on documented, held-out data.

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
   - Do not hardcode feature-importance claims; calculate and document them for each validated model artifact.

---

## Coding Standards & Guidelines

### Python & Machine Learning
- Prefer vectorized `pandas` / `numpy` operations over loops for data processing.
- Ensure all model training scripts maintain spatial/temporal cross-validation splits to prevent data leakage.
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
- **NO** fabricated accuracy claims. The 80% exceedance-recall value is a target, not a verified result. Report the metric, screening threshold, validation split, and event count together.
