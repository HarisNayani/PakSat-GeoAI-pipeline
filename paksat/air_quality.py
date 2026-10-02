"""Time-aware summaries for observed or model-estimated PM2.5 telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

MIN_24H_HOURLY_COVERAGE = 18
REQUIRED_OBSERVATION_COLUMNS = {"city", "timestamp", "pm25"}


def prepare_observation_upload(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validate uploaded city/time/PM2.5 observations and report excluded rows."""
    missing = REQUIRED_OBSERVATION_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Observation CSV is missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Observation CSV contains no data rows.")

    frame = data.copy()
    frame["city"] = frame["city"].astype("string").str.strip()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], format="mixed", errors="coerce", utc=True
    )
    frame["pm25"] = pd.to_numeric(frame["pm25"], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    valid = (
        frame["city"].notna()
        & frame["city"].ne("")
        & frame["timestamp"].notna()
        & frame["pm25"].notna()
        & frame["pm25"].ge(0)
    )
    excluded_rows = int((~valid).sum())
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows found. Check city names, timestamps, and non-negative PM2.5 values.")
    frame["record_source"] = "Uploaded historical data"
    return frame, excluded_rows


def filter_observation_period(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Return timestamped rows in an inclusive UTC calendar-date window."""
    if start_date > end_date:
        raise ValueError("Evidence period start must be on or before its end.")
    if "timestamp" not in data.columns:
        raise ValueError("Date filtering requires a `timestamp` column.")
    timestamps = pd.to_datetime(
        data["timestamp"], format="mixed", errors="coerce", utc=True
    )
    valid = timestamps.notna() & timestamps.dt.date.between(start_date, end_date)
    filtered = data.loc[valid].copy()
    filtered["timestamp"] = timestamps.loc[valid]
    return filtered


@dataclass(frozen=True)
class PM25ExposureSummary:
    latest_pm25: float
    latest_lower: float
    latest_upper: float
    latest_timestamp: pd.Timestamp | None
    trailing_24h_mean: float | None
    trailing_24h_hour_count: int
    latest_sensor_count: int | None


def summarize_city_pm25(data: pd.DataFrame, city: str) -> PM25ExposureSummary:
    """Summarize the latest city reading and a coverage-checked trailing mean.

    The 24-hour value is descriptive of available readings, not a forecast or a
    regulatory daily average. It is withheld unless at least 18 hourly buckets
    are represented in the trailing window.
    """
    if not {"city", "pm25"}.issubset(data.columns):
        raise ValueError("PM2.5 telemetry must contain `city` and `pm25` columns.")

    rows = data.loc[data["city"].astype(str).eq(city)].copy()
    rows["pm25"] = pd.to_numeric(rows["pm25"], errors="coerce")
    rows = rows.dropna(subset=["pm25"])
    if rows.empty:
        raise ValueError(f"No valid PM2.5 readings are available for {city}.")

    for column in ("pm25_lower", "pm25_upper"):
        if column not in rows:
            rows[column] = rows["pm25"]
        else:
            rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(rows["pm25"])

    if "timestamp" not in rows:
        timestamps = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    else:
        timestamps = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
    rows["_timestamp"] = timestamps
    timed_rows = rows.dropna(subset=["_timestamp"])

    if timed_rows.empty:
        latest_rows = rows
        latest_timestamp = None
        hour_count = 0
        trailing_mean = None
    else:
        latest_timestamp = timed_rows["_timestamp"].max()
        latest_rows = timed_rows.loc[timed_rows["_timestamp"].eq(latest_timestamp)]
        trailing_rows = timed_rows.loc[
            timed_rows["_timestamp"].gt(latest_timestamp - pd.Timedelta(hours=24))
            & timed_rows["_timestamp"].le(latest_timestamp)
        ].copy()
        trailing_rows["_hour"] = trailing_rows["_timestamp"].dt.floor("h")
        hourly_values = trailing_rows.groupby("_hour")["pm25"].median()
        hour_count = int(hourly_values.size)
        trailing_mean = (
            float(hourly_values.mean())
            if hour_count >= MIN_24H_HOURLY_COVERAGE
            else None
        )

    sensor_count = (
        int(latest_rows["sensor_id"].nunique())
        if "sensor_id" in latest_rows
        else None
    )
    return PM25ExposureSummary(
        latest_pm25=float(latest_rows["pm25"].median()),
        latest_lower=float(latest_rows["pm25_lower"].median()),
        latest_upper=float(latest_rows["pm25_upper"].median()),
        latest_timestamp=latest_timestamp,
        trailing_24h_mean=trailing_mean,
        trailing_24h_hour_count=hour_count,
        latest_sensor_count=sensor_count,
    )


__all__ = [
    "MIN_24H_HOURLY_COVERAGE",
    "PM25ExposureSummary",
    "filter_observation_period",
    "prepare_observation_upload",
    "summarize_city_pm25",
]