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
    radius_m: int = 25000


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
    """Fetch hourly PM2.5 means from nearby OpenAQ sensors."""
    if not api_key:
        return pd.DataFrame()

    headers = {"X-API-Key": api_key}
    radius_m = min(max(int(region.radius_m), 1), 25000)
    try:
        response = requests.get(
            "https://api.openaq.org/v3/locations",
            params={
                "coordinates": f"{region.latitude},{region.longitude}",
                "radius": radius_m,
                "parameters_id": 2,
                "limit": 1000,
            },
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        locations = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    seen_sensor_ids: set[int] = set()
    for location in locations:
        location_coordinates = location.get("coordinates") or {}
        for sensor in location.get("sensors", []):
            parameter = sensor.get("parameter") or {}
            sensor_id = sensor.get("id")
            if parameter.get("name", "").lower() != "pm25" or sensor_id is None:
                continue
            sensor_id = int(sensor_id)
            if sensor_id in seen_sensor_ids:
                continue
            seen_sensor_ids.add(sensor_id)
            try:
                response = requests.get(
                    f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
                    params={
                        "date_from": start_date.isoformat(),
                        "date_to": end_date.isoformat(),
                        "limit": 1000,
                    },
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                measurements = response.json().get("results", [])
            except (requests.RequestException, ValueError, TypeError):
                continue

            for measurement in measurements:
                period = measurement.get("period") or {}
                date_range = period.get("datetimeTo") or period.get("datetimeFrom") or {}
                coordinates = measurement.get("coordinates") or location_coordinates
                rows.append(
                    {
                        "city": region.name,
                        "timestamp": date_range.get("utc"),
                        "pm25": measurement.get("value"),
                        "latitude": coordinates.get("latitude", region.latitude),
                        "longitude": coordinates.get("longitude", region.longitude),
                        "location_id": location.get("id"),
                        "sensor_id": sensor_id,
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
