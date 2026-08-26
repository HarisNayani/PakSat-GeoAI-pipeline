"""Multi-sensor ingestion with optional Google Earth Engine and OpenAQ adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class CityRegion:
    name: str
    latitude: float
    longitude: float
    radius_m: int = 50000


PAKISTAN_REGIONS = (
    CityRegion("Lahore", 31.5204, 74.3587),
    CityRegion("Karachi", 24.8607, 67.0011),
    CityRegion("Islamabad", 33.6844, 73.0479),
)


def fetch_openaq_measurements(
    region: CityRegion,
    start_date: date,
    end_date: date,
    api_key: str | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch PM2.5 observations; return an empty frame on unavailable telemetry."""
    url = "https://api.openaq.org/v3/measurements"
    params: dict[str, Any] = {
        "coordinates": f"{region.latitude},{region.longitude}",
        "radius": region.radius_m,
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "parameter": "pm25",
        "limit": 1000,
    }
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        records = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return pd.DataFrame()

    rows = []
    for record in records:
        coordinates = record.get("coordinates") or {}
        rows.append(
            {
                "city": region.name,
                "timestamp": record.get("datetime", {}).get("utc"),
                "pm25": record.get("value"),
                "latitude": coordinates.get("latitude", region.latitude),
                "longitude": coordinates.get("longitude", region.longitude),
            }
        )
    return pd.DataFrame(rows)


def initialize_earth_engine() -> Any:
    """Initialize Earth Engine using native auth and an optional project variable."""
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install earthengine-api to enable satellite ingestion.") from exc

    try:
        ee.Initialize(project=os.getenv("GEE_PROJECT", "alibaba-hackathon-505817"))
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not authenticated. Run `earthengine authenticate` first."
        ) from exc
    return ee


def fetch_satellite_telemetry(
    region: CityRegion,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch satellite telemetry through a configured Earth Engine workflow.

    The collection reduction is intentionally isolated here so credentials and
    provider-specific query details never leak into feature or UI code.
    """
    ee = initialize_earth_engine()
    geometry = ee.Geometry.Point([region.longitude, region.latitude]).buffer(region.radius_m)
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterBounds(geometry)
        .filterDate(start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
        .select(["Optical_Depth_047"])
    )
    image = collection.mean()
    if image.bandNames().size().getInfo() == 0:
        return pd.DataFrame()
    value = image.reduceRegion(ee.Reducer.mean(), geometry, 1000).getInfo()
    return pd.DataFrame(
        [{"city": region.name, "AOD_047": value.get("Optical_Depth_047")}]
    )


def load_sensor_data(
    region: CityRegion,
    days: int = 1,
    include_satellite: bool = False,
) -> pd.DataFrame:
    """Combine available OpenAQ and optional satellite data for one region."""
    end_date = date.today()
    start_date = end_date - timedelta(days=max(days, 1))
    ground = fetch_openaq_measurements(
        region, start_date, end_date, api_key=os.getenv("OPENAQ_API_KEY")
    )
    if include_satellite:
        satellite = fetch_satellite_telemetry(region, start_date, end_date)
        if not satellite.empty:
            ground = ground.merge(satellite, on="city", how="left")
    return ground
