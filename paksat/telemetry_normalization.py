"""Normalize Earth Engine atmospheric covariates to model input units."""

from __future__ import annotations

from collections.abc import Mapping
from math import atan2, degrees, exp, hypot, isfinite


def _finite_number(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def normalize_earth_engine_features(
    raw_values: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Convert ERA5 Kelvin/u-v and MAIAC scaled AOD to model-ready units."""
    normalized = {
        key: _finite_number(value)
        for key, value in raw_values.items()
        if key not in {"temperature_kelvin", "dewpoint_kelvin", "AOD_047"}
    }

    aod = _finite_number(raw_values.get("AOD_047"))
    if aod is not None:
        normalized["AOD_047"] = aod * 0.001

    temperature_kelvin = _finite_number(raw_values.get("temperature_kelvin"))
    dewpoint_kelvin = _finite_number(raw_values.get("dewpoint_kelvin"))
    if temperature_kelvin is not None:
        temperature_celsius = temperature_kelvin - 273.15
        normalized["temperature"] = temperature_celsius
        if dewpoint_kelvin is not None:
            dewpoint_celsius = dewpoint_kelvin - 273.15
            actual_vapor = exp(17.625 * dewpoint_celsius / (243.04 + dewpoint_celsius))
            saturation_vapor = exp(17.625 * temperature_celsius / (243.04 + temperature_celsius))
            normalized["relative_humidity"] = min(max(actual_vapor / saturation_vapor, 0.0), 1.0)
    else:
        normalized["temperature"] = None
        normalized["relative_humidity"] = None

    eastward = _finite_number(raw_values.get("wind_u_mps"))
    northward = _finite_number(raw_values.get("wind_v_mps"))
    if eastward is not None and northward is not None:
        normalized["wind_speed"] = hypot(eastward, northward)
        normalized["wind_direction_degrees"] = (degrees(atan2(-eastward, -northward)) + 360.0) % 360.0
    else:
        normalized["wind_speed"] = None
        normalized["wind_direction_degrees"] = None

    return normalized


__all__ = ["normalize_earth_engine_features"]