"""Streamlit dashboard for air-quality monitoring and clinical decision support."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Element
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from cloud_ingestion import PAKISTAN_REGIONS, load_integrated_telemetry
from clinical_api_engine import assess_patient_risk, predict_hospital_capacity
from train_pai_model import predict_with_uncertainty

MODEL_PATH = Path(__file__).with_name("lightgbm_pm25_model.pkl")


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


def make_map(data: pd.DataFrame, latitude: float, longitude: float) -> folium.Map:
    # OpenStreetMap is public and does not require a tile API key.
    air_map = folium.Map(location=[latitude, longitude], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    heat_points = data[["latitude", "longitude", "pm25"]].dropna().values.tolist()
    if heat_points:
        HeatMap(heat_points, radius=28, blur=20, min_opacity=0.35).add_to(air_map)
    for row in data.dropna(subset=["latitude", "longitude", "pm25"]).itertuples():
        color = "red" if row.pm25 > 150 else "orange" if row.pm25 > 55 else "green"
        upper = getattr(row, "pm25_upper", row.pm25)
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=7, color=color, fill=True,
            fill_opacity=0.85, tooltip=f"{row.city}: {row.pm25:.1f} ug/m3 (upper {upper:.1f})",
        ).add_to(air_map)
    air_map.get_root().html.add_child(Element(
        '<div style="background:#fff;padding:8px 10px;border:1px solid #999;line-height:1.5">'
        '<b>PM2.5 (ug/m3)</b><br><span style="color:#16803c">&#9679;</span> &lt; 55 '
        '<span style="color:#f08c00">&#9679;</span> 55-150 '
        '<span style="color:#d62828">&#9679;</span> &gt; 150</div>'
    ))
    folium.Marker([latitude, longitude], tooltip="Selected location", icon=folium.Icon(color="blue")).add_to(air_map)
    return air_map


def main() -> None:
    st.set_page_config(page_title="PakSat EHI", page_icon="P", layout="wide")
    st.title("PakSat Air Quality Analyzer")
    st.caption("Environmental health intelligence for Pakistan")

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

    region = next(region for region in PAKISTAN_REGIONS if region.name == selected_city)
    artifact = load_model(str(MODEL_PATH))
    data = load_integrated_telemetry(region) if use_live else pd.DataFrame()
    if data.empty or "pm25" not in data.columns:
        data = demo_data()
        st.info("Showing local demonstration observations. Enable live telemetry when OpenAQ is available.")

    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")
    if artifact:
        try:
            predictions = predict_with_uncertainty(artifact, data)
            data["pm25"] = predictions["pm25_point"]
            data["pm25_upper"] = predictions["pm25_upper"]
        except (KeyError, ValueError, TypeError):
            st.info("Model features are incomplete for this observation; showing sensor PM2.5.")
    if "pm25_upper" not in data:
        data["pm25_upper"] = data["pm25"] * 1.15
    current = float(data.loc[data["city"].eq(selected_city), "pm25"].mean()) if data["city"].eq(selected_city).any() else float(data["pm25"].median())
    upper = float(data.loc[data["city"].eq(selected_city), "pm25_upper"].mean()) if data["city"].eq(selected_city).any() else float(data["pm25_upper"].median())
    risk = assess_patient_risk(current, upper, symptoms, {"age": age, "asthma": asthma, "copd": copd})
    surge = predict_hospital_capacity(current / 2.0, current / 100.0)

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


if __name__ == "__main__":
    main()
