"""PyDeck map construction for gridded PM2.5 and observed wind vectors."""

from __future__ import annotations

from math import cos, isfinite, radians, sin
from typing import Any

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


def build_pm25_deck(
    observations: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_PM25_THRESHOLDS,
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

    frame = observations.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
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
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )


__all__ = [
    "DEFAULT_PM25_THRESHOLDS",
    "PM25_COLORS",
    "PM25_THRESHOLD_PROFILES",
    "build_pm25_deck",
]