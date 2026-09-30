# PakSat: Environmental Health & Economic Intelligence

PakSat is a government decision-support pilot that connects neighborhood-scale air-pollution evidence to health-system pressure and transparent intervention economics. It is not another AQI display: the goal is to help officials compare where exposure is rising, what evidence supports that signal, which response to evaluate, and what locally sourced assumptions imply for public costs.

The current pilot covers Lahore, Karachi, and Islamabad. It is not yet a nationally representative estimate of pollution, health burden, or economic loss. Synthetic demo observations are clearly labelled and must never be presented as measured Pakistani outcomes.

## 🌟 Key Features
* **Physics-Informed Feature Engineering**: Models vertical aerosol compression (PBLH Inversion, Stagnation Index) and secondary photochemical smog kinetics.
* **Evaluation integrity**: Uses city-held-out `GroupKFold`; PM2.5 history features use prior observations only and are isolated by city. Because the model consumes historical ground PM2.5, this is a sensor-assisted nowcast, not satellite-only performance. Recompute and publish scores on a provenance-tracked dataset before making accuracy claims.
* **Clinical Risk Escalation**: Automated threshold escalation (`Environmental Acute Exacerbation` at $>150\,\mu\text{g/m}^3$) with 24-48h hospital admission surge warnings.
* **Decision evidence**: Regional PM2.5 distributions, record counts, provenance labels, and q05-q95 model bands only when a trained quantile artifact is available.
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

ERA5 temperature is normalized from Kelvin to Celsius; relative humidity is derived from temperature/dewpoint; wind speed and direction are derived from u/v components. Earth Engine satellite and weather values are currently regional means over the requested date window and are joined to ground rows by city, not by each sensor timestamp. This temporal mismatch must be resolved before interpreting the fused model as an hourly forecast or using it for causal policy evaluation. Synthetic demo values are not observations. The app reports no predictive interval for a point-only model artifact.

## 🚀 Quick Start
```bash
# 1. Clone & install dependencies
pip install -r requirements.txt

# 2. Optional: train on a validated CSV with `city`, `pm25`, and model features
python retrain_model.py path/to/validated_training_data.csv --output lightgbm_pm25_model.pkl

# 3. Launch Streamlit UI (works in labelled demo mode without a model artifact)
streamlit run app.py
```

Live OpenAQ ground data requires an `OPENAQ_API_KEY` environment variable; the dashboard remains available in labelled demo mode without it. Earth Engine covariates require `earthengine authenticate` and a valid `GEE_PROJECT`. Never place either credential in source code. Run the test suite with `python -m unittest discover -s tests -v`.

## Enterprise Modules

Install the additional libraries with the rest of the project dependencies using `pip install -r requirements.txt`. The Streamlit dashboard has tabs for 3D geospatial visualization, causal policy analysis, and resource allocation; the existing Folium monitoring view remains available.

### Causal policy panel

The policy tab provides a CSV schema template. Supply at least 10 complete, aligned observations containing all of these columns:

| Column | Meaning |
| --- | --- |
| `brick_kiln_shutdown_intensity` | Normalized policy intensity in `[0, 1]` |
| `crop_burning_penalties_intensity` | Normalized policy intensity in `[0, 1]` |
| `heavy_vehicle_restriction_intensity` | Normalized policy intensity in `[0, 1]` |
| `ground_emissions` | Ground-emissions measurement or validated proxy |
| `pblh_inversion` | Boundary-layer inversion indicator/measurement |
| `stagnation_index` | Atmospheric stagnation index |
| `pm25` | Neighborhood PM2.5 outcome in µg/m³ |
| `respiratory_admissions` | Admissions outcome with consistent population and time interval |
| `policy_context` | Measured policy targeting/context confounder |
| `meteorology_index` | Pre-specified meteorological confounder/index |

`CausalPolicySimulator(panel).simulate_policy_impact(policy_type, intensity)` estimates outcome changes relative to zero intensity. `simulate_policy_impact(policy_type, intensity, simulator=...)` is also available as a module-level function. Effects are not hard-coded. The graph assumes supplied context and meteorology variables adequately address confounding. Validate time alignment, treatment overlap/positivity, measurement quality, and identification assumptions before interpreting estimates as causal or making policy decisions. Admissions changes retain the outcome units and observation interval in the panel.

The policy tab's PKR calculation is a separate scenario, not part of DoWhy. It requires the user to provide a source for cost per admission, workdays lost, and value per workday. The result is per panel observation only; it is not annualized, weighted to a national population, or presented as realized savings. Review possible overlap between medical costs and productivity losses.

### Geospatial observations

`build_pm25_deck(data)` requires numeric `latitude`, `longitude`, and `pm25` columns. Wind vectors are included when data has `wind_u_mps`/`wind_v_mps`, or `wind_speed` plus `wind_direction_degrees` (meteorological direction from which wind blows). The map offers WHO 24-hour guideline/interim-target concentrations and common EPA AQI PM2.5 breakpoints. Confirm EPA breakpoint adoption with the relevant Pakistan EPA authority before production. Demo wind directions are illustrative only.

### Resource optimization

`optimize_hospital_resources(predicted_surge_data, available_supplies)` accepts per-hospital demand keyed by `emergency_beds`, `oxygen_cylinders`, and `nebulizer_stations`, with an optional positive `risk_weight`. Supplies are system-wide counts. The LP maximizes severity-weighted fulfilled demand, subject to supply and demand bounds. Results may be fractional; apply integer rounding and local clinical/logistics constraints before deployment. Dashboard scenario values are editable examples, not live inventories.
