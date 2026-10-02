"""Streamlit dashboard for air-quality monitoring and clinical decision support."""

from __future__ import annotations

import logging
import os
import pickle
from hashlib import sha256
from datetime import date
from pathlib import Path
from typing import Any, cast

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Element
from dotenv import load_dotenv
from streamlit_folium import st_folium

from cloud_ingestion import PAKISTAN_REGIONS, load_integrated_telemetry
from clinical_api_engine import assess_patient_risk
from paksat.causal_policy import (
    PANEL_ALIGNMENT_COLUMNS,
    PANEL_COLUMNS,
    CausalPolicySimulator,
    simulate_policy_impact,
)
from paksat.economic_impact import EconomicAssumptions, estimate_health_economic_benefit
from paksat.air_quality import (
    MIN_24H_HOURLY_COVERAGE,
    filter_observation_period,
    prepare_observation_upload,
    summarize_city_pm25,
)
from paksat.geospatial import (
    PAKISTAN_BOUNDS,
    PM25_THRESHOLD_PROFILES,
    build_pm25_deck,
    prepare_pm25_map_data,
)
from paksat.resource_optimizer import RESOURCE_TYPES, optimize_hospital_resources
from pipeline import PM25_EXCEEDANCE_RECALL_TARGET
from train_pai_model import predict_with_uncertainty

MODEL_PATH = Path(__file__).with_name("lightgbm_pm25_model.pkl")
LOGGER = logging.getLogger(__name__)
load_dotenv(Path(__file__).with_name(".env"), override=False)


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


@st.cache_data(ttl=300, max_entries=4, show_spinner=False)
def cached_pm25_map_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cache bounded spatial bins, not a large raw sensor payload."""
    return prepare_pm25_map_data(data)


def make_map(data: pd.DataFrame, latitude: float, longitude: float) -> folium.Map:
    # OpenStreetMap is public and does not require a tile API key.
    air_map = folium.Map(location=[latitude, longitude], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    map_data = data.copy()
    for column in ("latitude", "longitude", "pm25", "pm25_upper"):
        if column in map_data.columns:
            map_data[column] = pd.to_numeric(map_data[column], errors="coerce")
    map_data = map_data.dropna(subset=["latitude", "longitude", "pm25"])
    map_data = map_data.loc[
        map_data["latitude"].between(*PAKISTAN_BOUNDS["latitude"])
        & map_data["longitude"].between(*PAKISTAN_BOUNDS["longitude"])
    ]
    if len(map_data) > 1000:
        map_data = map_data.sample(n=1000, random_state=42)
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
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        color="#1769aa",
        fill=True,
        fill_color="#1769aa",
        fill_opacity=1.0,
        tooltip="Selected location",
    ).add_to(air_map)
    return air_map


def main() -> None:
    st.set_page_config(page_title="Zameen | Environmental Health Intelligence", page_icon="Z", layout="wide")
    st.title("Zameen | Environmental Health Intelligence")
    st.caption(
        "Decision support for hospitals and government: connect air-pollution "
        "evidence, clinical preparedness, and transparent intervention scenarios. "
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
        historical_file = st.file_uploader(
            "Upload historical observations (CSV)",
            type=["csv"],
            help="Required columns: city, timestamp, pm25. Optional: latitude, longitude, sensor_id, weather and satellite features. Valid uploads replace demo/live rows.",
            key="historical_observations",
        )
        st.download_button(
            "Download observation CSV template",
            data=pd.DataFrame(
                columns=["city", "timestamp", "pm25", "latitude", "longitude", "sensor_id"]
            ).to_csv(index=False),
            file_name="zameen_observations_template.csv",
            mime="text/csv",
        )
        uploaded_observations: pd.DataFrame | None = None
        excluded_upload_rows = 0
        if historical_file is not None:
            try:
                uploaded_observations, excluded_upload_rows = prepare_observation_upload(
                    pd.read_csv(historical_file)
                )
                if excluded_upload_rows:
                    st.warning(f"Excluded {excluded_upload_rows:,} invalid upload rows.")
            except (ValueError, pd.errors.ParserError, UnicodeError) as error:
                st.error(f"Historical upload could not be used: {error}")
        city_options = [region.name for region in PAKISTAN_REGIONS]
        if uploaded_observations is not None:
            city_options = list(dict.fromkeys([
                *city_options,
                *uploaded_observations["city"].astype(str).unique().tolist(),
            ]))
        selected_city = st.selectbox("Sensor region", city_options)
        use_live = st.checkbox(
            "Use live OpenAQ telemetry",
            value=False,
            disabled=uploaded_observations is not None,
        )

    artifact = load_model(str(MODEL_PATH))
    live_unavailable: list[str] = []
    model_uncertainty_available = False
    live_status_message: str | None = None
    has_openaq_key = bool(os.getenv("OPENAQ_API_KEY"))
    if uploaded_observations is not None:
        data = uploaded_observations.copy()
    elif use_live:
        if has_openaq_key:
            with st.spinner("Loading available regional telemetry..."):
                data, live_unavailable = load_live_regional_telemetry()
        else:
            data = pd.DataFrame()
            live_status_message = (
                "Live OpenAQ credentials are not configured. This demo remains fully "
                "usable with clearly labeled synthetic observations."
            )
        if "pm25" in data.columns:
            data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            data = data.dropna(subset=["pm25"])
            data = data.loc[data["pm25"] >= 0]
        if not data.empty and "city" in data.columns and not data["city"].astype(str).eq(selected_city).any():
            selected_demo = demo_data().loc[lambda rows: rows["city"].eq(selected_city)].copy()
            selected_demo["record_source"] = "Synthetic demo fallback"
            data = pd.concat([data, selected_demo], sort=False).reset_index(drop=True)
            live_status_message = (
                f"{selected_city} has no live observations; its panel uses labeled demo "
                "data. Other available regions remain live."
            )
    else:
        data = pd.DataFrame()

    if (
        data.empty
        or "pm25" not in data.columns
        or not pd.to_numeric(data["pm25"], errors="coerce").ge(0).any()
    ):
        data = demo_data()
        data["record_source"] = "Synthetic demo"
        if use_live and has_openaq_key:
            unavailable_text = ", ".join(live_unavailable) if live_unavailable else "all regions"
            live_status_message = (
                f"Live telemetry is currently unavailable for {unavailable_text}. "
                "Showing clearly labeled synthetic demo observations instead."
            )
        elif not use_live:
            live_status_message = "Demo mode: showing clearly labeled synthetic observations."
    elif "record_source" not in data.columns:
        data["record_source"] = "Live telemetry"

    if uploaded_observations is not None:
        st.success(
            f"Historical upload active: {len(data):,} valid observations. "
            "Uploaded data is kept separate from demo and live telemetry."
        )
    if live_status_message:
        if use_live and uploaded_observations is None:
            st.warning(live_status_message)
        else:
            st.info(live_status_message)

    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    data = data.dropna(subset=["pm25"])
    data = data.loc[data["pm25"] >= 0]
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

    if "timestamp" in data.columns:
        available_timestamps = pd.to_datetime(
            data["timestamp"], format="mixed", errors="coerce", utc=True
        )
        available_dates = available_timestamps.dropna().dt.date
        if not available_dates.empty:
            earliest_date, latest_date = available_dates.min(), available_dates.max()
            selected_period = st.date_input(
                "Evidence date range (UTC)",
                value=(earliest_date, latest_date),
                min_value=earliest_date,
                max_value=latest_date,
                help="The evidence table, percentile summary and trend use this inclusive date range.",
                key="evidence_date_range",
            )
            if isinstance(selected_period, date):
                start_date = end_date = selected_period
            elif isinstance(selected_period, (tuple, list)) and len(selected_period) == 2:
                start_date = cast(date, selected_period[0])
                end_date = cast(date, selected_period[1])
            else:
                st.error("Choose a valid start and end date.")
                st.stop()
            data = filter_observation_period(data, start_date, end_date)
            if data.empty:
                st.warning("No observations fall inside this date range. Expand the range to continue.")
                st.stop()
            st.caption(
                f"Evidence window: {start_date.isoformat()} to {end_date.isoformat()} UTC "
                f"({len(data):,} observations)."
            )
        else:
            st.info("No valid timestamps are available for a historical trend. Upload rows with city, timestamp and PM2.5.")
    else:
        st.info("Historical trends require timestamped observations. Upload a CSV to choose an evidence date range.")

    selected_observations = data.loc[data["city"].astype(str).eq(selected_city)]
    if selected_observations.empty:
        selected_observations = data
        st.info(f"No {selected_city} observations fall in this period; showing {selected_observations.iloc[0]['city']} instead.")
        selected_city = str(selected_observations.iloc[0]["city"])
    exposure = summarize_city_pm25(data, selected_city)
    current = exposure.latest_pm25
    upper = exposure.latest_upper
    risk = assess_patient_risk(current, upper, symptoms, {"age": age, "asthma": asthma, "copd": copd})
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
    reading_label = "Latest" if exposure.latest_timestamp is not None else "Window median"
    evidence_cols[2].metric(
        f"{selected_city} {reading_label} PM2.5",
        f"{current:.1f} µg/m³",
    )
    if model_uncertainty_available:
        evidence_cols[3].metric(
            "Latest model q05-q95 band",
            f"{exposure.latest_lower:.1f} to {upper:.1f} µg/m³",
        )
    else:
        average_text = (
            f"{exposure.trailing_24h_mean:.1f} µg/m³"
            if exposure.trailing_24h_mean is not None
            else f"Insufficient ({exposure.trailing_24h_hour_count}/{MIN_24H_HOURLY_COVERAGE} hours)"
        )
        evidence_cols[3].metric("Trailing 24-hour mean", average_text)
    if exposure.latest_timestamp is not None:
        sensor_text = (
            f" across {exposure.latest_sensor_count} sensors"
            if exposure.latest_sensor_count is not None
            else ""
        )
        st.caption(
            f"Latest {selected_city} sample: {exposure.latest_timestamp.isoformat()}"
            f"{sensor_text}. The 24-hour mean uses hourly medians and is shown only "
            f"with at least {MIN_24H_HOURLY_COVERAGE} distinct hourly readings."
        )
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
                st.info("This evidence window has fewer than two distinct timestamps, so a trend cannot be plotted.")
        st.caption(
            "P95 and the trend use valid rows inside the selected date range. "
            "Uploaded and synthetic demo observations are not regulatory 24-hour averages."
        )
    st.caption(
        "The current pilot covers Lahore, Karachi, and Islamabad. Economic values below "
        "are per panel observation and require locally sourced assumptions; no national "
        "annual total is inferred from this sample."
    )
    with st.expander("Model validation and 80% target"):
        variant = artifact.get("model_variant") if artifact else None
        validation_metrics = (artifact or {}).get("metrics", {}).get(variant, {})
        if validation_metrics:
            recall = validation_metrics.get("pm25_exceedance_recall", float("nan"))
            threshold = validation_metrics.get("pm25_exceedance_threshold_ug_m3")
            if threshold is not None and np.isfinite(recall):
                status = "Target met" if recall >= PM25_EXCEEDANCE_RECALL_TARGET else "Below target"
                st.metric(
                    "Held-out exceedance recall",
                    f"{recall:.0%}",
                    delta=status,
                    help="Recall of held-out rows whose observed PM2.5 meets or exceeds the configured screening cutoff.",
                )
                st.caption(
                    f"Goal: at least {PM25_EXCEEDANCE_RECALL_TARGET:.0%}. "
                    f"Cutoff: {threshold:g} µg/m³. Validation rows: "
                    f"{int(validation_metrics.get('rows', 0)):,}. "
                    "This is row-level exceedance recall, not generic accuracy or a validated forecast. "
                    "Confirm the cutoff and averaging period with local authorities."
                )
            else:
                st.info("The validation set had no exceedance events, so recall cannot be calculated.")
            if "rmse_ug_m3" in validation_metrics:
                st.caption(f"Held-out PM2.5 RMSE: {validation_metrics['rmse_ug_m3']:.1f} µg/m³.")
        else:
            st.info("No saved model validation metrics are available. Train on validated, timestamped observations to assess the 80% target.")

    monitoring_tab, map_tab, policy_tab, resources_tab = st.tabs(
        ["Monitoring & triage", "3D geospatial", "Policy simulator", "Resource allocation"]
    )

    with monitoring_tab:
        st.subheader(f"{selected_city} monitoring")
        metric_cols = st.columns(4)
        metric_cols[0].metric(f"{reading_label} PM2.5", f"{current:.1f} μg/m³")
        metric_cols[1].metric("Upper bound", f"{upper:.1f} μg/m³")
        metric_cols[2].metric("Risk level", risk["severity"])
        metric_cols[3].metric("Hospital forecast", "Not connected")

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
            st.subheader("Capacity planning")
            st.info("No validated 24-48h hospital-demand forecast is connected.")
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
            map_data = cached_pm25_map_data(data)
            st.pydeck_chart(build_pm25_deck(map_data, thresholds=thresholds, aggregated=True))
            if not ({"wind_u_mps", "wind_v_mps"}.issubset(data.columns) or "wind_direction_degrees" in data.columns):
                st.info("Wind vectors are not available in these observations; the flow overlay appears when wind direction or u/v components are ingested.")
        except (RuntimeError, ValueError) as error:
            st.warning(f"3D map unavailable: {error}")
        st.caption(f"{threshold_profile} breakpoints (µg/m³): {thresholds}")

    with policy_tab:
        st.subheader("Causal policy intervention simulator")
        st.warning("Observational counterfactual scenarios are not forecasts or identified causal effects. Validate assumptions and local policy data before decision use.")
        policy_file = st.file_uploader(
            "Upload aligned policy and outcome panel (CSV)",
            type=["csv"],
            key="policy_panel",
            help="Requires all columns documented in README.md, including intervention intensity, mediators, outcomes, and confounders.",
        )
        st.download_button(
            "Download policy-panel CSV template",
            data=pd.DataFrame(
                columns=[*PANEL_ALIGNMENT_COLUMNS, *sorted(PANEL_COLUMNS)]
            ).to_csv(index=False),
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
                    result_cols[0].metric("Scenario PM2.5 change", f"{estimate['estimated_pm25_change_ug_m3']:+.2f} µg/m³")
                    result_cols[1].metric("Scenario admissions change per observation", f"{estimate['estimated_admissions_change']:+.2f}")
                    st.caption(estimate["interpretation"])
                    if estimate.get("diagnostics"):
                        st.warning("Model diagnostic: " + " ".join(estimate["diagnostics"]))
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
        scenario_demand_data: dict[str, dict[str, float]] = {}
        hospital_columns = st.columns(len(hospital_defaults))
        for (hospital, defaults), column in zip(hospital_defaults.items(), hospital_columns):
            with column:
                st.markdown(f"**{hospital}**")
                scenario_demand_data[hospital] = {
                    "risk_weight": float(st.number_input(
                        "Priority weight", min_value=0.1, max_value=5.0, value=1.0,
                        step=0.1, key=f"priority_{hospital}",
                        help="Relative planning priority only; agree weights with hospital operations and clinical leadership.",
                    )),
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
                allocation = optimize_hospital_resources(scenario_demand_data, available_supplies)
                result_columns = st.columns(2)
                with result_columns[0]:
                    st.markdown("**Allocated resources**")
                    st.dataframe(pd.DataFrame(allocation["allocation"]).T, width="stretch")
                with result_columns[1]:
                    st.markdown("**Unmet demand**")
                    st.dataframe(pd.DataFrame(allocation["unmet_demand"]).T, width="stretch")
                st.metric("Weighted unmet critical-care risk", f"{allocation['total_unmet_critical_care_risk']:.2f}")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
