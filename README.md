# 🛰️ PakSat: Environmental Health Intelligence & Clinical Triage Platform

PakSat is a GeoAI platform engineered for high-accuracy ground-level PM2.5 estimation across Pakistan (Lahore, Karachi, Islamabad) using satellite remote sensing, weather reanalysis, and gradient boosted models. The output directly powers an AI-assisted hospital triage engine for respiratory emergency load forecasting.

## 🌟 Key Features
* **Physics-Informed Feature Engineering**: Models vertical aerosol compression (PBLH Inversion, Stagnation Index) and secondary photochemical smog kinetics.
* **Spatial Leakage Prevention**: Trained using `GroupKFold` spatial cross-validation across regional test splits ($R^2 \approx 0.945$, $\text{RMSE} \approx 9.91\,\mu\text{g/m}^3$).
* **Clinical Risk Escalation**: Automated threshold escalation (`Environmental Acute Exacerbation` at $>150\,\mu\text{g/m}^3$) with 24-48h hospital admission surge warnings.
* **Streamlit & Folium GIS Interface**: Real-time raster map overlay and clinical patient portal.

## 🚀 Quick Start
```bash
# 1. Clone & install dependencies
pip install -r requirements.txt

# 2. Train model & run spatial cross-validation
python retrain_model.py

# 3. Launch Streamlit UI
streamlit run app.py
