"""Streamlit dashboard for air-quality monitoring and clinical decision support."""

from __future__ import annotations

import logging
import os
import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Element
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from cloud_ingestion import PAKISTAN_REGIONS, load_integrated_telemetry
from clinical_api_engine import assess_patient_risk, predict_hospital_capacity
from paksat.causal_policy import PANEL_COLUMNS, CausalPolicySimulator, simulate_policy_impact
from paksat.economic_impact import EconomicAssumptions, estimate_health_economic_benefit
from paksat.geospatial import PM25_THRESHOLD_PROFILES, build_pm25_deck
from paksat.resource_optimizer import RESOURCE_TYPES, optimize_hospital_resources
from train_pai_model import predict_with_uncertainty

MODEL_PATH = Path(__file__).with_name("lightgbm_pm25_model.pkl")
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def load_model(path: str) -> dict[str, Any] | None:
    """Load a trained artifact once; a missing artifact enables demo mode."""
    model_file = Path(path)
    if not model_file.exists():
        return None
    with model_file.open("rb") as source:
        artifact = pickle.load(source)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Model artifact must contain a `model` entry.")
    return artifact


@st.cache_data
def demo_data() -> pd.DataFrame:
    """Provide a transparent local fallback while remote sensors are unavailable."""
    rows = [
        {
            "city": "Lahore",
            "timestamp": "2026-09-03T10:00:00Z",
            "latitude": 31.5204,
            "longitude": 74.3587,
            "pm25": 118.0,
            "AOD_047": 0.92,
            "NO2_density": 0.72,
            "temperature": 31.5,
            "relative_humidity": 0.72,
            "wind_speed": 2.1,
            "wind_direction_degrees": 315.0,
            "pblh": 480.0,
            "atmospheric_stagnation_index": 1000.0 / (2.1 * 480.0 + 1.0),
            "thermal_confinement_ratio": 31.5 / (480.0 / 100.0),
            "photochemical_pm25_proxy": 0.72 * 31.5 * 0.92,
            "hygroscopic_growth_factor": 1.0 / (1.0 - 0.72),
            "lag_24h": 86.0,
            "lag_48h": 78.0,
        },
        {
            "city": "Karachi",
            "timestamp": "2026-09-03T10:00:00Z",
            "latitude": 24.8607,
            "longitude": 67.0011,
            "pm25": 74.0,
            "AOD_047": 0.48,
            "NO2_density": 0.38,
            "temperature": 30.2,
            "relative_humidity": 0.67,
            "wind_speed": 4.2,
            "wind_direction_degrees": 230.0,
            "pblh": 680.0,
            "atmospheric_stagnation_index": 1000.0 / (4.2 * 680.0 + 1.0),
            "thermal_confinement_ratio": 30.2 / (680.0 / 100.0),
            "photochemical_pm25_proxy": 0.38 * 30.2 * 0.48,
            "hygroscopic_growth_factor": 1.0 / (1.0 - 0.67),
            "lag_24h": 63.0,
            "lag_48h": 58.0,
        },
        {
            "city": "Islamabad",
            "timestamp": "2026-09-03T10:00:00Z",
            "latitude": 33.6844,
            "longitude": 73.0479,
            "pm25": 41.0,
            "AOD_047": 0.32,
            "NO2_density": 0.24,
            "temperature": 27.8,
            "relative_humidity": 0.55,
            "wind_speed": 6.1,
            "wind_direction_degrees": 45.0,
            "pblh": 860.0,
            "atmospheric_stagnation_index": 1000.0 / (6.1 * 860.0 + 1.0),
            "thermal_confinement_ratio": 27.8 / (860.0 / 100.0),
            "photochemical_pm25_proxy": 0.24 * 27.8 * 0.32,
            "hygroscopic_growth_factor": 1.0 / (1.0 - 0.55),
            "lag_24h": 37.0,
            "lag_48h": 32.0,
        },
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def load_live_regional_telemetry() -> tuple[pd.DataFrame, list[str]]:
    """Load and cache available telemetry for the three current pilot regions."""
    frames: list[pd.DataFrame] = []
    unavailable: list[str] = []
    for region in PAKISTAN_REGIONS:
        try:
            observations = load_integrated_telemetry(region)
        except Exception:
            LOGGER.exception("Telemetry load failed for %s", region.name)
            unavailable.append(region.name)
            continue
        if observations.empty or "pm25" not in observations.columns:
            unavailable.append(region.name)
            continue
        observations = observations.copy()
        observations["record_source"] = "Live telemetry"
        frames.append(observations)
    if not frames:
        return pd.DataFrame(), unavailable
    return pd.concat(frames, ignore_index=True, sort=False), unavailable


def make_map(data: pd.DataFrame, latitude: float, longitude: float) -> folium.Map:
    # OpenStreetMap is public and does not require a tile API key.
    air_map = folium.Map(location=[latitude, longitude], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    map_data = data.copy()
    for column in ("latitude", "longitude", "pm25", "pm25_upper"):
        if column in map_data.columns:
            map_data[column] = pd.to_numeric(map_data[column], errors="coerce")
    map_data = map_data.dropna(subset=["latitude", "longitude", "pm25"])
    heat_points = map_data[["latitude", "longitude", "pm25"]].values.tolist()
    if heat_points:
        HeatMap(heat_points, radius=28, blur=20, min_opacity=0.35).add_to(air_map)
    for raw_row in map_data.to_dict(orient="records"):
        row = cast(dict[str, Any], raw_row)
        row_pm25 = float(row["pm25"])
        color = "red" if row_pm25 > 150 else "orange" if row_pm25 > 55 else "green"
        upper = float(row.get("pm25_upper", row_pm25))
        city = str(row.get("city", "Unknown location"))
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])], radius=7, color=color, fill=True,
            fill_opacity=0.85, tooltip=f"{city}: {row_pm25:.1f} ug/m3 (upper {upper:.1f})",
        ).add_to(air_map)
    air_map.get_root().add_child(Element(
        '<div style="background:#fff;padding:8px 10px;border:1px solid #999;line-height:1.5">'
        '<b>PM2.5 (ug/m3)</b><br><span style="color:#16803c">&#9679;</span> &lt; 55 '
        '<span style="color:#f08c00">&#9679;</span> 55-150 '
        '<span style="color:#d62828">&#9679;</span> &gt; 150</div>'
    ))
    folium.Marker([latitude, longitude], tooltip="Selected location", icon=folium.Icon(color="blue")).add_to(air_map)
    return air_map


def main() -> None:
    st.set_page_config(page_title="PakSat EHI", page_icon="P", layout="wide")
    st.title("PakSat | Public Health & Economic Intelligence")
    st.caption(
        "A three-city government decision-support pilot connecting air-pollution "
        "evidence, clinical pressure, and transparent intervention economics. "
        "Not an official AQI bulletin or a national burden estimate."
    )

    with st.sidebar:
        st.header("Patient context")
        latitude = st.number_input("Latitude", min_value=23.5, max_value=37.5, value=31.5204, format="%.4f")
        longitude = st.number_input("Longitude", min_value=60.0, max_value=78.0, value=74.3587, format="%.4f")
        symptoms = st.multiselect("Symptoms", ["Cough", "Wheezing", "Shortness of breath", "Chest tightness", "Severe breathing difficulty", "Blue lips"])
        age = st.number_input("Patient age", min_value=0, max_value=120, value=35)
        asthma = st.checkbox("Asthma history")
        copd = st.checkbox("COPD history")
        selected_city = st.selectbox("Sensor region", [region.name for region in PAKISTAN_REGIONS])
        use_live = st.checkbox("Use live OpenAQ telemetry", value=False)

    artifact = load_model(str(MODEL_PATH))
    live_unavailable: list[str] = []
    model_uncertainty_available = False
    if use_live:
        if not os.getenv("OPENAQ_API_KEY"):
            st.warning("Set OPENAQ_API_KEY to fetch live ground PM2.5 from OpenAQ; without it the dashboard uses clearly labelled synthetic demo data.")
        with st.spinner("Loading available regional telemetry..."):
            data, live_unavailable = load_live_regional_telemetry()
        if "pm25" in data.columns:
            data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")
            data = data.dropna(subset=["pm25"])
        if not data.empty and "city" in data.columns and not data["city"].astype(str).eq(selected_city).any():
            selected_demo = demo_data().loc[lambda rows: rows["city"].eq(selected_city)].copy()
            selected_demo["record_source"] = "Synthetic demo fallback"
            data = pd.concat([data, selected_demo], ignore_index=True, sort=False)
            st.warning(
                f"No live PM2.5 rows are available for {selected_city}; its displayed "
                "record is synthetic. Other available regional rows remain live."
            )
    else:
        data = pd.DataFrame()

    if data.empty or "pm25" not in data.columns or not data["pm25"].notna().any():
        data = demo_data()
        data["record_source"] = "Synthetic demo"
        st.info("Showing illustrative demo observations. Enable live telemetry for regional sensor data.")
    elif "record_source" not in data.columns:
        data["record_source"] = "Live telemetry"

    if use_live and live_unavailable:
        st.warning(f"Live data is unavailable for: {', '.join(live_unavailable)}. Those regions are not represented as measured data.")

    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")
    if artifact:
        try:
            predictions = predict_with_uncertainty(artifact, data)
            data["pm25"] = predictions["pm25_point"]
            data["pm25_lower"] = predictions["pm25_lower"]
            data["pm25_upper"] = predictions["pm25_upper"]
            model_uncertainty_available = bool(
                np.isfinite(predictions["pm25_lower"].to_numpy(dtype=float)).all()
                and np.isfinite(predictions["pm25_upper"].to_numpy(dtype=float)).all()
            )
        except Exception:
            LOGGER.exception("PM2.5 model inference failed")
            st.warning("Model inference failed; showing measured/demo PM2.5 without a predictive interval.")
    if not model_uncertainty_available:
        data["pm25_lower"] = data["pm25"]
        data["pm25_upper"] = data["pm25"]

    selected_observations = data.loc[data["city"].astype(str).eq(selected_city)]
    if selected_observations.empty:
        selected_observations = data
    current = float(selected_observations["pm25"].median())
    upper = float(selected_observations["pm25_upper"].median())
    risk = assess_patient_risk(current, upper, symptoms, {"age": age, "asthma": asthma, "copd": copd})
    surge = predict_hospital_capacity(current / 2.0, current / 100.0)

    st.subheader("Government evidence brief")
    summary = (
        data.groupby("city", as_index=False)
        .agg(
            observations=("pm25", "count"),
            median_pm25_ug_m3=("pm25", "median"),
            p95_pm25_ug_m3=("pm25", lambda values: float(values.quantile(0.95))),
            data_source=("record_source", "first"),
        )
        .sort_values("median_pm25_ug_m3", ascending=False)
    )
    evidence_cols = st.columns(4)
    evidence_cols[0].metric("Regions represented", str(data["city"].nunique()))
    evidence_cols[1].metric("PM2.5 records", f"{int(data['pm25'].count()):,}")
    evidence_cols[2].metric(f"{selected_city} median PM2.5", f"{current:.1f} µg/m³")
    if model_uncertainty_available:
        lower = float(selected_observations["pm25_lower"].median())
        evidence_cols[3].metric("Model q05-q95 band", f"{lower:.1f} to {upper:.1f} µg/m³")
    else:
        evidence_cols[3].metric("Predictive interval", "Unavailable")
    with st.expander("Regional evidence ledger", expanded=True):
        st.dataframe(summary, hide_index=True)
        st.download_button(
            "Download evidence CSV",
            data=data.to_csv(index=False),
            file_name="paksat_regional_evidence.csv",
            mime="text/csv",
        )
        if "timestamp" in data.columns:
            trend = data[["timestamp", "city", "pm25"]].copy()
            trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce", utc=True)
            trend = trend.dropna(subset=["timestamp", "pm25"])
            if trend["timestamp"].nunique() > 1:
                trend_table = trend.pivot_table(
                    index="timestamp",
                    columns="city",
                    values="pm25",
                    aggfunc="median",
                ).sort_index()
                st.line_chart(trend_table)
            else:
                st.info("A time trend needs multiple valid timestamps; no trend is inferred from a single snapshot.")
        st.caption(
            "P95 is descriptive of the rows currently loaded. Demo rows are synthetic; "
            "sensor-level observations are not regulatory 24-hour averages."
        )
    st.caption(
        "The current pilot covers Lahore, Karachi, and Islamabad. Economic values below "
        "are per panel observation and require locally sourced assumptions; no national "
        "annual total is inferred from this sample."
    )

    monitoring_tab, map_tab, policy_tab, resources_tab = st.tabs(
        ["Monitoring & triage", "3D geospatial", "Policy simulator", "Resource allocation"]
    )

    with monitoring_tab:
        st.subheader(f"{selected_city} nowcast")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Current PM2.5", f"{current:.1f} μg/m³")
        metric_cols[1].metric("Upper bound", f"{upper:.1f} μg/m³")
        metric_cols[2].metric("Risk level", risk["severity"])
        metric_cols[3].metric("24-48h capacity", f"{surge['estimated_surge_percent']:.1f}%")

        col_map, col_triage = st.columns([1.4, 1])
        with col_map:
            st.subheader("Regional PM2.5")
            st_folium(make_map(data, latitude, longitude), width=None, height=520)
            with st.expander("AI feature drivers"):
                city_row = data.loc[data["city"] == selected_city].iloc[0] if (data["city"] == selected_city).any() else data.iloc[0]
                st.write({
                    "AOD_047": round(float(city_row.get("AOD_047", 0.0)), 3),
                    "NO2_density": round(float(city_row.get("NO2_density", 0.0)), 3),
                    "temperature_C": round(float(city_row.get("temperature", 0.0)), 1),
                    "relative_humidity": round(float(city_row.get("relative_humidity", 0.0)), 3),
                    "wind_speed_mps": round(float(city_row.get("wind_speed", 0.0)), 2),
                    "pblh_m": round(float(city_row.get("pblh", 0.0)), 1),
                })
        with col_triage:
            st.subheader("Clinical triage")
            if risk["urgent"]:
                st.error(risk["severity"])
            else:
                st.warning(risk["severity"])
            for recommendation in risk["advice"]:
                st.write(f"- {recommendation}")
            st.subheader("Emergency capacity")
            if surge["capacity_warning"]:
                st.warning(surge["message"])
            else:
                st.success(surge["message"])
            st.caption(risk["disclaimer"])

    with map_tab:
        st.subheader("PM2.5 concentration and wind")
        threshold_profile = st.selectbox("PM2.5 color thresholds", list(PM25_THRESHOLD_PROFILES))
        thresholds = PM25_THRESHOLD_PROFILES[threshold_profile]
        st.caption(
            "Columns show concentration-scaled PM2.5 across six green-to-maroon "
            "bands. Confirm EPA breakpoint adoption locally before operational use."
        )
        try:
            st.pydeck_chart(build_pm25_deck(data, thresholds=thresholds))
            if not ({"wind_u_mps", "wind_v_mps"}.issubset(data.columns) or "wind_direction_degrees" in data.columns):
                st.info("Wind vectors are not available in these observations; the flow overlay appears when wind direction or u/v components are ingested.")
        except (RuntimeError, ValueError) as error:
            st.warning(f"3D map unavailable: {error}")
        st.caption(f"{threshold_profile} breakpoints (µg/m³): {thresholds}")

    with policy_tab:
        st.subheader("Causal policy intervention simulator")
        st.warning("Observational estimates are not forecasts. Validate identification assumptions and local policy data before using results for decisions.")
        policy_file = st.file_uploader(
            "Upload aligned policy and outcome panel (CSV)",
            type=["csv"],
            key="policy_panel",
            help="Requires all columns documented in README.md, including intervention intensity, mediators, outcomes, and confounders.",
        )
        st.download_button(
            "Download policy-panel CSV template",
            data=pd.DataFrame(columns=sorted(PANEL_COLUMNS)).to_csv(index=False),
            file_name="paksat_policy_panel_template.csv",
            mime="text/csv",
        )
        selected_policy = st.selectbox(
            "Policy intervention",
            ["brick_kiln_shutdown", "crop_burning_penalties", "heavy_vehicle_restriction"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        policy_intensity = st.slider("Intervention intensity", 0.0, 1.0, 0.5, 0.05)
        if policy_file is None:
            st.session_state.pop("policy_impact_estimate", None)
            st.info("Upload at least 10 complete, aligned observations to estimate policy effects. No default effect sizes are fabricated.")
        else:
            try:
                policy_panel = pd.read_csv(policy_file)
                simulator = CausalPolicySimulator(policy_panel)
                panel_fingerprint = sha256(policy_file.getvalue()).hexdigest()
                if st.button("Estimate intervention effect", key="simulate_policy"):
                    estimate = simulate_policy_impact(selected_policy, policy_intensity, simulator=simulator)
                    st.session_state["policy_impact_estimate"] = {
                        "estimate": estimate,
                        "policy_type": selected_policy,
                        "intensity": policy_intensity,
                        "panel_fingerprint": panel_fingerprint,
                    }
                saved_estimate = st.session_state.get("policy_impact_estimate")
                if (
                    saved_estimate
                    and saved_estimate["policy_type"] == selected_policy
                    and saved_estimate["intensity"] == policy_intensity
                    and saved_estimate["panel_fingerprint"] == panel_fingerprint
                ):
                    estimate = saved_estimate["estimate"]
                    result_cols = st.columns(2)
                    result_cols[0].metric("Estimated PM2.5 reduction", f"{estimate['estimated_pm25_reduction_ug_m3']:.2f} µg/m³")
                    result_cols[1].metric("Projected admissions reduction per panel observation", f"{estimate['projected_admissions_reduction']:.2f}")
                    st.caption(estimate["interpretation"])
                    with st.expander("Translate to an auditable PKR scenario"):
                        medical_cost = st.number_input("Direct medical cost per admission (PKR)", min_value=0.0, value=0.0, step=1000.0)
                        workdays_lost = st.number_input("Workdays lost per admission", min_value=0.0, value=0.0, step=0.5)
                        daily_value = st.number_input("Value per workday (PKR)", min_value=0.0, value=0.0, step=500.0)
                        assumption_source = st.text_input("Source for these economic assumptions")
                        reporting_period = st.text_input("Panel observation unit (for example, city-day)", value="per panel observation")
                        if assumption_source.strip():
                            economic_value = estimate_health_economic_benefit(
                                estimate["projected_admissions_reduction"],
                                EconomicAssumptions(
                                    medical_cost_per_admission_pkr=medical_cost,
                                    workdays_lost_per_admission=workdays_lost,
                                    value_per_workday_pkr=daily_value,
                                    assumption_source=assumption_source,
                                    reporting_period=reporting_period,
                                ),
                            )
                            st.metric(
                                f"Scenario value ({reporting_period})",
                                f"PKR {economic_value['total_scenario_value_pkr']:,.0f}",
                            )
                            st.caption(
                                f"Medical: PKR {economic_value['direct_medical_savings_pkr']:,.0f}; "
                                f"productivity: PKR {economic_value['productivity_value_pkr']:,.0f}. "
                                f"Assumption source: {economic_value['assumption_source']}. "
                                f"{economic_value['disclaimer']}"
                            )
                        else:
                            st.info("Enter a source before interpreting this scenario as an economic estimate.")
                elif saved_estimate:
                    st.info("Policy inputs or panel changed. Run the estimate again to refresh results.")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))

    with resources_tab:
        st.subheader("Regional respiratory-care resource allocation")
        st.caption("Scenario inputs below are editable planning assumptions, not live hospital inventories or clinical forecasts.")
        hospital_defaults = {
            "Mayo Hospital Lahore": (12, 18, 8),
            "JPMC Karachi": (10, 16, 7),
            "PIMS Islamabad": (8, 12, 6),
        }
        predicted_surge_data: dict[str, dict[str, float]] = {}
        hospital_columns = st.columns(len(hospital_defaults))
        for (hospital, defaults), column in zip(hospital_defaults.items(), hospital_columns):
            with column:
                st.markdown(f"**{hospital}**")
                predicted_surge_data[hospital] = {
                    "emergency_beds": float(st.number_input("Bed demand", min_value=0, value=defaults[0], key=f"beds_{hospital}")),
                    "oxygen_cylinders": float(st.number_input("Oxygen cylinders", min_value=0, value=defaults[1], key=f"oxygen_{hospital}")),
                    "nebulizer_stations": float(st.number_input("Nebulizer stations", min_value=0, value=defaults[2], key=f"nebulizers_{hospital}")),
                }
        supply_columns = st.columns(len(RESOURCE_TYPES))
        supply_labels = {
            "emergency_beds": "Available emergency beds",
            "oxygen_cylinders": "Available oxygen cylinders",
            "nebulizer_stations": "Available nebulizer stations",
        }
        available_supplies = {
            resource: float(supply_columns[index].number_input(
                supply_labels[resource], min_value=0, value=defaults, key=f"supply_{resource}"
            ))
            for index, (resource, defaults) in enumerate(zip(RESOURCE_TYPES, (20, 30, 12)))
        }
        if st.button("Optimize allocation", key="optimize_resources"):
            try:
                allocation = optimize_hospital_resources(predicted_surge_data, available_supplies)
                st.dataframe(pd.DataFrame(allocation["allocation"]).T)
                st.metric("Weighted unmet critical-care risk", f"{allocation['total_unmet_critical_care_risk']:.2f}")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
