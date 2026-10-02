"""Cloud-aware telemetry ingestion for the PakSat platform.

All provider SDKs are optional at import time. The dashboard can therefore run in
local demo mode while production deployments inject Alibaba Cloud and Earth
Engine credentials through environment variables.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping

import pandas as pd

from data_ingestion import (
    CityRegion,
    PAKISTAN_REGIONS,
    fetch_openaq_measurements,
    merge_prior_covariates,
)
from paksat.telemetry_normalization import normalize_earth_engine_features


@dataclass(frozen=True)
class OSSConfig:
    """Alibaba Cloud OSS connection settings."""

    bucket: str
    endpoint: str
    access_key_id: str
    access_key_secret: str

    @classmethod
    def from_env(cls) -> "OSSConfig | None":
        values = {
            "bucket": os.getenv("ALIYUN_OSS_BUCKET"),
            "endpoint": os.getenv("ALIYUN_OSS_ENDPOINT"),
            "access_key_id": os.getenv("ALIYUN_OSS_ACCESS_KEY_ID"),
            "access_key_secret": os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET"),
        }
        return cls(**values) if all(values.values()) else None


class OSSCache:
    """Small OSS adapter for binary NetCDF and raster tile caching."""

    def __init__(self, config: OSSConfig | None = None) -> None:
        self.config = config or OSSConfig.from_env()
        self._bucket: Any | None = None
        if self.config:
            try:
                import oss2
                auth = oss2.Auth(self.config.access_key_id, self.config.access_key_secret)
                self._bucket = oss2.Bucket(auth, self.config.endpoint, self.config.bucket)
            except ImportError:
                self._bucket = None

    @property
    def available(self) -> bool:
        return self._bucket is not None

    def get(self, key: str) -> bytes | None:
        """Read an object, returning ``None`` for unavailable or missing cache."""
        if not self._bucket:
            return None
        try:
            return self._bucket.get_object(key).read()
        except Exception:
            return None

    def put(self, key: str, payload: bytes) -> bool:
        """Write an object and report whether the provider accepted it."""
        if not self._bucket:
            return False
        try:
            self._bucket.put_object(key, payload)
            return True
        except Exception:
            return False


class AnalyticDBPostGIS:
    """Parameterized AnalyticDB for PostgreSQL/PostGIS station index adapter."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("ANALYTICDB_POSTGRES_DSN")

    def upsert_stations(self, frame: pd.DataFrame, table: str = "openaq_stations") -> int:
        """Upsert station observations when a DSN and psycopg are configured."""
        required = {"city", "latitude", "longitude", "pm25", "timestamp"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing station columns: {sorted(missing)}")
        if not self.dsn or frame.empty:
            return 0
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("Table name must be a simple PostgreSQL identifier.")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] for AnalyticDB writes.") from exc
        statement = f"""INSERT INTO {table} (city, latitude, longitude, pm25, observed_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (city, observed_at) DO UPDATE SET pm25 = EXCLUDED.pm25"""
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                columns = ["city", "latitude", "longitude", "pm25", "timestamp"]
                cursor.executemany(statement, frame[columns].itertuples(index=False, name=None))
        return len(frame)


def _openaq(region: CityRegion, start: date, end: date) -> pd.DataFrame:
    return fetch_openaq_measurements(
        region,
        start,
        end,
        api_key=os.getenv("OPENAQ_API_KEY"),
    )


def _earth_engine(region: CityRegion, start: date, end: date) -> pd.DataFrame:
    """Reduce MODIS AOD, Sentinel-5P NO2, and ERA5 at a regional centroid."""
    try:
        import ee
        ee.Initialize(project=os.getenv("GEE_PROJECT", "alibaba-hackathon-505817"))
        geometry = ee.Geometry.Point([region.longitude, region.latitude]).buffer(region.radius_m)
        end_exclusive = (end + timedelta(days=1)).isoformat()
        collections = {
            "AOD_047": ("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047"),
            "NO2_density": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
            "temperature_kelvin": ("ECMWF/ERA5_LAND/HOURLY", "temperature_2m"),
            "dewpoint_kelvin": ("ECMWF/ERA5_LAND/HOURLY", "dewpoint_temperature_2m"),
            "wind_u_mps": ("ECMWF/ERA5_LAND/HOURLY", "u_component_of_wind_10m"),
            "wind_v_mps": ("ECMWF/ERA5_LAND/HOURLY", "v_component_of_wind_10m"),
            "pblh": ("ECMWF/ERA5/HOURLY", "boundary_layer_height"),
        }
        values: dict[str, float | None] = {}
        for output, (collection_id, band) in collections.items():
            image = ee.ImageCollection(collection_id).filterBounds(geometry).filterDate(start.isoformat(), end_exclusive).select(band).mean()
            values[output] = image.reduceRegion(ee.Reducer.mean(), geometry, 1000).getInfo().get(band)
        normalized = normalize_earth_engine_features(values)
        return pd.DataFrame([{"city": region.name, **normalized}])
    except Exception:
        return pd.DataFrame()


def load_integrated_telemetry(region: CityRegion, days: int = 2, cache: OSSCache | None = None) -> pd.DataFrame:
    """Load ground data and only temporally safe satellite/weather covariates."""
    end = date.today()
    start = end - timedelta(days=max(days, 1))
    ground = _openaq(region, start, end)
    satellite = _earth_engine(region, start, end)
    if not satellite.empty:
        ground = merge_prior_covariates(ground, satellite)
        # Untimestamped regional averages cannot be safely joined to historical
        # ground rows without introducing look-ahead bias.
    if ground.empty:
        return ground
    # Do not interpolate sensor targets or covariates across observation times;
    # model-side imputers are fitted on training folds only.
    if cache and cache.available:
        cache.put(f"telemetry/{region.name.lower()}-{end.isoformat()}.csv", ground.to_csv(index=False).encode())
    return ground


__all__ = ["OSSCache", "OSSConfig", "AnalyticDBPostGIS", "CityRegion", "PAKISTAN_REGIONS", "load_integrated_telemetry"]
