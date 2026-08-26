"""Streamlit dashboard for air-quality monitoring and clinical decision support."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from data_ingestion import PAKISTAN_REGIONS, CityRegion, load_sensor_data
from generate_enhanced_features import add_physics_features, prepare_features
from triage_engine import assess_clinical_risk, predict_hospital_surge

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


def demo_data() -> pd.DataFrame:
    """Provide a transparent local fallback while remote sensors are unavailable."""
    return pd.DataFrame(
        [
            {"city": "Lahore", "latitude": 31.5204, "longitude": 74.3587, "pm25": 118.0, "AOD_047": 0.9, "NO2_density": 0.7},
            {"city": "Karachi", "latitude": 24.8607, "longitude": 67.0011, "pm25": 72.0, "AOD_047": 0.5, "NO2_density": 0.4},
            {"city": "Islamabad", "latitude": 33.6844, "longitude": 73.0479, "pm25": 42.0, "AOD_047": 0.3, "NO2_density": 0.25},
        ]
    )


def make_map(data: pd.DataFrame, latitude: float, longitude: float) -> folium.Map:
    air_map = folium.Map(location=[latitude, longitude], zoom_start=5, tiles="CartoDB positron")
    heat_points = data[["latitude", "longitude", "pm25"]].dropna().values.tolist()
    if heat_points:
        HeatMap(heat_points, radius=28, blur=20, min_opacity=0.35).add_to(air_map)
    for row in data.dropna(subset=["latitude", "longitude", "pm25"]).itertuples():
        color = "red" if row.pm25 > 150 else "orange" if row.pm25 > 55 else "green"
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=7, color=color, fill=True,
            fill_opacity=0.85, tooltip=f"{row.city}: {row.pm25:.1f} ug/m3",
        ).add_to(air_map)
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
        selected_city = st.selectbox("Sensor region", [region.name for region in PAKISTAN_REGIONS])
        use_live = st.checkbox("Use live OpenAQ telemetry", value=False)

    region = next(region for region in PAKISTAN_REGIONS if region.name == selected_city)
    data = load_sensor_data(region) if use_live else pd.DataFrame()
    if data.empty:
        data = demo_data()
        st.info("Showing local demonstration observations. Enable live telemetry when OpenAQ is available.")

    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")
    current = float(data.loc[data["city"].eq(selected_city), "pm25"].mean()) if data["city"].eq(selected_city).any() else float(data["pm25"].median())
    col_map, col_triage = st.columns([1.4, 1])
    with col_map:
        st.subheader("Regional PM2.5")
        st_folium(make_map(data, latitude, longitude), width=None, height=520)
    with col_triage:
        st.subheader("Clinical triage")
        risk = assess_clinical_risk(current, symptoms)
        st.metric("Estimated PM2.5", f"{current:.1f} ug/m3")
        st.warning(risk["severity"] if risk["urgent"] else risk["severity"])
        for recommendation in risk["recommendations"]:
            st.write(f"- {recommendation}")
        surge = predict_hospital_surge(current / 2.0, current / 100.0)
        st.subheader("Emergency capacity")
        st.metric("24-48 hour surge", f"{surge['estimated_surge_percent']:.1f}%")
        if surge["capacity_warning"]:
            st.warning(surge["message"])
        else:
            st.success(surge["message"])
        st.caption(risk["disclaimer"])


if __name__ == "__main__":
    main()
