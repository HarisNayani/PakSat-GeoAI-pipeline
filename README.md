# PakSat: Environmental Health & Economic Intelligence

PakSat is a government decision-support pilot that connects neighborhood-scale air-pollution evidence to health-system pressure and transparent intervention economics. It is not another AQI display: the goal is to help officials compare where exposure is rising, what evidence supports that signal, which response to evaluate, and what locally sourced assumptions imply for public costs.

The current pilot covers Lahore, Karachi, and Islamabad. It is not yet a nationally representative estimate of pollution, health burden, or economic loss. Synthetic demo observations are clearly labelled and must never be presented as measured Pakistani outcomes.

## 🌟 Key Features
* **Physics-Informed Feature Engineering**: Models vertical aerosol compression (PBLH Inversion, Stagnation Index) and secondary photochemical smog kinetics.
* **Evaluation integrity**: Reports separate forward-time sensor-assisted nowcasting and city-held-out satellite/meteorology/coordinate downscaling scores. Learned median imputation is fit within each training fold. Publish per-city/per-time results on a provenance-tracked dataset before making accuracy claims.
* **Clinical Risk Escalation**: Symptom- and exposure-based decision support; no validated hospital-demand forecast is currently connected.
* **Decision evidence**: Regional PM2.5 distributions, record counts, and provenance labels.
* **Time-aware monitoring**: Latest timestamped PM2.5 readings and a trailing 24-hour mean only when at least 18 distinct hourly readings are available; this is descriptive, not a forecast or regulatory daily average.
* **Geospatial analysis**: Folium monitoring plus a PyDeck 3D PM2.5 layer and measured-wind vector overlay.
* **Causal policy scenarios**: DoWhy estimates for brick kiln shutdowns, crop-burning penalties, and heavy-vehicle restrictions from an uploaded, aligned observational panel.
* **Health-system readiness**: Clinical escalation and a SciPy allocation optimizer for beds, oxygen cylinders, and nebulizer stations.
* **Economic translation**: Convert estimated avoided admissions into per-observation PKR scenarios using explicit, source-labelled local cost and productivity inputs. The app does not annualize or extrapolate pilot results to Pakistan.

## Government Decision Loop

1. **Observe**: Combine ground observations with available satellite and meteorological signals; inspect coverage and source labels.
2. **Prioritize**: Compare regional PM2.5 evidence and clinical pressure instead of relying on a single AQI score.
3. **Evaluate**: Estimate intervention effects only when a suitable policy/outcome panel is supplied; otherwise, do not show fabricated policy impacts.
4. **Value**: Translate an estimated admissions effect into a bounded PKR scenario with cited local assumptions and the panel's time unit.
5. **Act and verify**: Allocate resources, record the intervention, and measure subsequent outcomes against a documented baseline.

The app is a decision-support prototype, not a regulatory monitor, causal proof, or substitute for clinical judgement. A credible deployment needs validated monitoring networks, city-level population and health data, policy timing/enforcement records, and independently reviewed economic assumptions.

## Data Provenance and Limits

* Ground PM2.5: [OpenAQ measurements API](https://docs.openaq.org/).
* Aerosol optical depth: [NASA MODIS MCD19A2 MAIAC](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES); `Optical_Depth_047` is converted from its stored scale to physical AOD.
* Nitrogen dioxide: [Sentinel-5P OFFL L3 NO2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2).
* Meteorology: [ERA5-Land hourly](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY) for temperature, dewpoint, and 10 m u/v winds; [ERA5 hourly](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY) for boundary-layer height.

ERA5 temperature is normalized from Kelvin to Celsius; relative humidity is derived from temperature/dewpoint; wind speed and direction are derived from u/v components. Current Earth Engine results are regional means without a matching sensor timestamp, so live ingestion withholds them rather than joining them to historical ground rows. Timestamped covariates are joined backward in time only, with a six-hour tolerance. The model refuses satellite predictions without row-level AOD, coordinates, and meteorological support. Training files still require independent source and time-alignment verification. Synthetic demo values are not observations.

The dashboard's trailing 24-hour PM2.5 summary takes the mean of hourly city-level medians and is shown only when at least 18 distinct hourly buckets are present. It summarizes available telemetry; it is not a validated regulatory 24-hour average or a future smog forecast.

## 🚀 Quick Start
```bash
# 1. Clone & install dependencies
pip install -r requirements.txt

# 2. Optional: train on a validated CSV with `city`, `pm25`, and model features
python retrain_model.py path/to/validated_training_data.csv --output lightgbm_pm25_model.pkl

# 3. Launch Streamlit UI (works in labelled demo mode without a model artifact)
streamlit run app.py
```

Live OpenAQ ground data requires an `OPENAQ_API_KEY` environment variable; the dashboard loads it from a local, git-ignored `.env` file when present. The dashboard starts in clearly labelled demo mode so the presentation does not depend on external API availability; enable live telemetry from the sidebar when desired. Earth Engine covariates require `earthengine authenticate` and a valid `GEE_PROJECT`. Never place credentials in source code or commit them. Run the test suite with `python -m unittest discover -s tests -v`.

### Upload historical observations

With the dashboard running, open `http://localhost:8502` and use **Upload historical observations (CSV)** in the sidebar. Download the adjacent CSV template for the expected headers. The required columns are `city`, `timestamp`, and `pm25`; timestamps must be parseable, and PM2.5 values must be numeric and non-negative. Optional columns include `latitude`, `longitude`, and `sensor_id`; supported model features may also be supplied. Invalid rows are excluded with a count, and a valid upload replaces demo/live observations for that dashboard session. Use the UTC date-range control to scope the evidence table, P95 summaries, and trend chart. The app does not persist the uploaded file.

## Enterprise Modules

Install the additional libraries with the rest of the project dependencies using `pip install -r requirements.txt`. The Streamlit dashboard has tabs for 3D geospatial visualization, causal policy analysis, and resource allocation; the existing Folium monitoring view remains available.

### Causal policy panel

The policy tab provides a CSV schema template. Supply at least 10 complete, aligned observations containing all of these columns:

| Column | Meaning |
| --- | --- |
| `city` | City identifier for the observation |
| `timestamp` | Parseable observation time; city/timestamp pairs must be unique |
| `brick_kiln_shutdown_intensity` | Normalized policy intensity in `[0, 1]` |
| `crop_burning_penalties_intensity` | Normalized policy intensity in `[0, 1]` |
| `heavy_vehicle_restriction_intensity` | Normalized policy intensity in `[0, 1]` |
| `ground_emissions` | Ground-emissions measurement or validated proxy |
| `pblh` | Planetary boundary-layer height in metres |
| `stagnation_index` | Atmospheric stagnation index |
| `wind_speed` | Wind speed in metres per second |
| `pm25` | Neighborhood PM2.5 outcome in µg/m³ |
| `respiratory_admissions` | Admissions outcome with consistent population and time interval |
| `policy_context` | Measured policy targeting/context confounder |

`CausalPolicySimulator(panel).simulate_policy_impact(policy_type, intensity)` returns an **observational counterfactual scenario** relative to zero intensity. `simulate_policy_impact(policy_type, intensity, simulator=...)` is also available as a module-level function. The engine requires valid city/timestamp keys, rejects duplicate city-time rows and treatment doses outside observed support, and requires near-zero controls and dose overlap. These checks do not prove identification; validate panel cadence, confounding, measurement quality, and uncertainty before policy use. Admissions changes retain the outcome units and observation interval in the panel.

The policy tab's PKR calculation is a separate scenario, not part of DoWhy. It requires the user to provide a source for cost per admission, workdays lost, and value per workday. The result is per panel observation only; it is not annualized, weighted to a national population, or presented as realized savings. Review possible overlap between medical costs and productivity losses.

### Geospatial observations

`build_pm25_deck(data)` requires numeric `latitude`, `longitude`, and `pm25` columns. Wind vectors are included when data has `wind_u_mps`/`wind_v_mps`, or `wind_speed` plus `wind_direction_degrees` (meteorological direction from which wind blows). The map offers WHO 24-hour guideline/interim-target concentrations and common EPA AQI PM2.5 breakpoints. Confirm EPA breakpoint adoption with the relevant Pakistan EPA authority before production. Demo wind directions are illustrative only.

### Resource optimization

`optimize_hospital_resources(predicted_surge_data, available_supplies)` accepts integer per-hospital demand keyed by `emergency_beds`, `oxygen_cylinders`, and `nebulizer_stations`, with an optional positive `risk_weight`. Supplies are system-wide integer counts. The MILP maximizes severity-weighted fulfilled demand, subject to supply and demand bounds. Dashboard scenario values are editable planning assumptions, not live inventories or a validated 24-48h demand forecast.

### Model training

Run `python train_pai_model.py path/to/validated_training_data.csv`. The saved artifact contains both a sensor-assisted model and an unmonitored model. Sensor-assisted validation is forward-time and computes station lags only from training-period history; unmonitored evaluation holds out city/spatial groups and excludes NO2 and ground-PM2.5 history. Validation reports PM2.5 RMSE and row-level recall for readings at or above a configurable screening cutoff (`--pm25-exceedance-threshold`, default 55 µg/m³). The 80% recall value is a project target, not a verified result; confirm the cutoff and averaging period with local authorities. An older artifact without the new variant/schema must be retrained rather than passed through the legacy whole-frame feature path.
