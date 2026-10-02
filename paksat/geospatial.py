"""PyDeck map construction for gridded PM2.5 and observed wind vectors."""

from __future__ import annotations

from math import cos, isfinite, radians, sin
from typing import Any

import numpy as np
import pandas as pd

PM25_THRESHOLD_PROFILES = {
    "WHO 24-hour AQG/IT": (15.0, 25.0, 37.5, 50.0, 75.0),
    "EPA AQI PM2.5 (confirm local adoption)": (12.0, 35.4, 55.4, 150.4, 250.4),
}
DEFAULT_PM25_THRESHOLDS = PM25_THRESHOLD_PROFILES["WHO 24-hour AQG/IT"]
PM25_COLORS = (
    (35, 139, 69, 190),
    (255, 205, 0, 200),
    (245, 130, 32, 210),
    (210, 45, 45, 220),
    (125, 60, 152, 225),
    (128, 0, 32, 230),
)
PAKISTAN_BOUNDS = {"latitude": (23.5, 37.5), "longitude": (60.0, 78.0)}
DEFAULT_GRID_DEGREES = 0.05
MAX_MAP_POINTS = 5000


def _color_for_pm25(value: float, thresholds: tuple[float, ...]) -> list[int]:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return list(PM25_COLORS[index])
    return list(PM25_COLORS[-1])


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _wind_paths(data: pd.DataFrame) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    has_components = {"wind_u_mps", "wind_v_mps"}.issubset(data.columns)
    has_direction = {"wind_speed", "wind_direction_degrees"}.issubset(data.columns)
    if not has_components and not has_direction:
        return paths

    for row in data.to_dict(orient="records"):
        latitude = _finite_float(row.get("latitude"))
        longitude = _finite_float(row.get("longitude"))
        if latitude is None or longitude is None:
            continue
        if has_components:
            eastward = _finite_float(row.get("wind_u_mps"))
            northward = _finite_float(row.get("wind_v_mps"))
        else:
            eastward = northward = None
        if (eastward is None or northward is None) and has_direction:
            speed = _finite_float(row.get("wind_speed"))
            direction = _finite_float(row.get("wind_direction_degrees"))
            if speed is None or direction is None:
                continue
            toward = radians(direction + 180.0)
            eastward, northward = speed * sin(toward), speed * cos(toward)
        if eastward is None or northward is None:
            continue
        longitude_scale = max(cos(radians(latitude)), 0.1)
        end = [longitude + eastward * 0.01 / longitude_scale, latitude + northward * 0.01]
        paths.append({"path": [[longitude, latitude], end], "color": [35, 86, 145, 220], "width": 3})
    return paths


def prepare_pm25_map_data(
    observations: pd.DataFrame,
    *,
    grid_degrees: float = DEFAULT_GRID_DEGREES,
    max_points: int = MAX_MAP_POINTS,
) -> pd.DataFrame:
    """Validate Pakistan-domain points, keep latest cell readings, then bin/cap rows."""
    required = {"latitude", "longitude", "pm25"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Map observations are missing columns: {sorted(missing)}")
    if not isfinite(grid_degrees) or grid_degrees <= 0 or max_points < 1:
        raise ValueError("Grid size and map payload limit must be positive.")

    frame = observations.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame.loc[
        frame["latitude"].between(*PAKISTAN_BOUNDS["latitude"])
        & frame["longitude"].between(*PAKISTAN_BOUNDS["longitude"])
        & frame["pm25"].ge(0.0)
    ].copy()
    if frame.empty:
        raise ValueError("No valid PM2.5 observations within the Pakistan map bounds.")

    has_components = {"wind_u_mps", "wind_v_mps"}.issubset(frame.columns)
    has_direction = {"wind_speed", "wind_direction_degrees"}.issubset(frame.columns)
    if has_components:
        frame["wind_u_mps"] = pd.to_numeric(frame["wind_u_mps"], errors="coerce")
        frame["wind_v_mps"] = pd.to_numeric(frame["wind_v_mps"], errors="coerce")
    else:
        frame["wind_u_mps"] = np.nan
        frame["wind_v_mps"] = np.nan
    if has_direction:
        speed = pd.to_numeric(frame["wind_speed"], errors="coerce")
        direction = pd.to_numeric(frame["wind_direction_degrees"], errors="coerce")
        toward = np.radians(direction + 180.0)
        frame["wind_u_mps"] = frame["wind_u_mps"].fillna(speed * np.sin(toward))
        frame["wind_v_mps"] = frame["wind_v_mps"].fillna(speed * np.cos(toward))

    frame["_lat_bin"] = np.floor(
        (frame["latitude"] - PAKISTAN_BOUNDS["latitude"][0]) / grid_degrees + 1e-9
    ).astype(int)
    frame["_lon_bin"] = np.floor(
        (frame["longitude"] - PAKISTAN_BOUNDS["longitude"][0]) / grid_degrees + 1e-9
    ).astype(int)
    bins = ["_lat_bin", "_lon_bin"]

    # Map the latest available PM2.5 per cell instead of mixing days into one value.
    if "timestamp" in frame:
        frame["_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if frame["_timestamp"].notna().any():
            frame = frame.dropna(subset=["_timestamp"])
            latest = frame.groupby(bins)["_timestamp"].transform("max")
            frame = frame.loc[frame["_timestamp"].eq(latest)]

    aggregated = (
        frame.groupby(bins, as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            pm25=("pm25", "mean"),
            wind_u_mps=("wind_u_mps", "mean"),
            wind_v_mps=("wind_v_mps", "mean"),
        )
        .drop(columns=bins)
    )
    if len(aggregated) > max_points:
        # A deterministic cap bounds the serialized WebGL payload across reruns.
        aggregated = aggregated.sample(n=max_points, random_state=42).sort_index()
    return aggregated.reset_index(drop=True)


def build_pm25_deck(
    observations: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_PM25_THRESHOLDS,
    aggregated: bool = False,
    max_points: int = MAX_MAP_POINTS,
) -> Any:
    """Build a 3D PM2.5 ColumnLayer with an optional observed wind PathLayer.

    The WHO profile uses its 24-hour guideline and interim targets. The EPA
    profile uses common EPA AQI PM2.5 concentration breakpoints; confirm local
    Pakistan EPA adoption and pass approved breakpoints through ``thresholds``.
    Wind vectors are rendered only when components or direction measurements
    are present in the source observations.
    """
    required = {"latitude", "longitude", "pm25"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Map observations are missing columns: {sorted(missing)}")
    if len(thresholds) != len(PM25_COLORS) - 1 or any(
        not isfinite(float(value)) for value in thresholds
    ) or tuple(sorted(thresholds)) != thresholds:
        raise ValueError("Thresholds must be five finite, ascending PM2.5 breakpoints.")

    try:
        import pydeck as pdk
    except ImportError as error:
        raise RuntimeError("Install the `pydeck` dependency to render the 3D map.") from error

    frame = observations.copy() if aggregated else prepare_pm25_map_data(
        observations, max_points=max_points
    )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame.loc[
        frame["latitude"].between(*PAKISTAN_BOUNDS["latitude"])
        & frame["longitude"].between(*PAKISTAN_BOUNDS["longitude"])
    ].head(max_points)
    if frame.empty:
        raise ValueError("No valid PM2.5 grid observations are available to map.")

    frame["elevation"] = frame["pm25"].clip(lower=0.0) * 12.0
    frame["color"] = frame["pm25"].map(lambda value: _color_for_pm25(float(value), thresholds))
    layers = [
        pdk.Layer(
            "ColumnLayer",
            data=frame,
            get_position="[longitude, latitude]",
            get_elevation="elevation",
            get_fill_color="color",
            radius=4500,
            elevation_scale=1,
            extruded=True,
            pickable=True,
            auto_highlight=True,
        )
    ]
    winds = _wind_paths(frame)
    if winds:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=winds,
                get_path="path",
                get_color="color",
                get_width="width",
                width_min_pixels=2,
                pickable=True,
            )
        )

    center_lat = float(frame["latitude"].mean())
    center_lon = float(frame["longitude"].mean())
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=4.5,
            pitch=48,
            bearing=0,
        ),
        tooltip={"html": "<b>PM2.5:</b> {pm25} µg/m³", "style": {"color": "white"}},
        map_style=None,
    )


__all__ = [
    "DEFAULT_PM25_THRESHOLDS",
    "PM25_COLORS",
    "PM25_THRESHOLD_PROFILES",
    "PAKISTAN_BOUNDS",
    "prepare_pm25_map_data",
    "build_pm25_deck",
]