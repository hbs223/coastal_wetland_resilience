"""Small shared utilities for paths, single-band rasters, and output files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def load_config(path: str | Path) -> dict:
    """Load a JSON configuration; resolve relative paths beside that file."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    years = config["years"]
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be nonempty and must not contain duplicates.")
    if any(not isinstance(year, int) for year in years):
        raise ValueError("Each assessment year must be an integer.")
    config["_base_dir"] = path.parent
    return config


def resolve_path(config: dict, value: str) -> Path:
    """Resolve an absolute path or a path relative to the configuration file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_base_dir"]) / path
    return path.resolve()


def ensure_new_directory(path: str | Path) -> Path:
    """Create an output directory without overwriting an earlier analysis."""
    path = Path(path)
    if path.exists() and any(item.name != ".gitkeep" for item in path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Choose a new output directory in the configuration."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a single-band GeoTIFF as float32, mapping declared NoData to NaN."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input raster not found: {path}")
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected one raster band: {path}")
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    metadata = {
        "shape": data.shape,
        "transform": dataset.GetGeoTransform(),
        "projection": dataset.GetProjection(),
    }
    if not metadata["projection"]:
        raise ValueError(f"Raster has no coordinate reference system: {path}")
    dataset = None
    return data, metadata


def assert_same_grid(reference: dict, candidate: dict, label: str) -> None:
    """Reject rasters with different shapes, pixel grids, or coordinate systems."""
    if tuple(reference["shape"]) != tuple(candidate["shape"]):
        raise ValueError(f"Raster dimensions differ: {label}")
    if not np.allclose(
        reference["transform"], candidate["transform"], rtol=0, atol=1e-9
    ):
        raise ValueError(f"Raster alignment differs: {label}")
    first, second = osr.SpatialReference(), osr.SpatialReference()
    first.ImportFromWkt(reference["projection"])
    second.ImportFromWkt(candidate["projection"])
    if not first.IsSame(second):
        raise ValueError(f"Coordinate reference systems differ: {label}")


def write_raster(path: str | Path, data: np.ndarray, metadata: dict) -> None:
    """Write a float32 GeoTIFF with explicit NoData and preserved georeferencing."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing raster: {path}")
    if data.shape != tuple(metadata["shape"]):
        raise ValueError("Output array dimensions differ from the reference grid.")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.where(np.isfinite(data), data, -9999).astype(np.float32)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    dataset.SetGeoTransform(metadata["transform"])
    dataset.SetProjection(metadata["projection"])
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(values)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def write_json(path: str | Path, data) -> None:
    """Write readable UTF-8 metadata; nonfinite numbers are not valid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, default=_json_default, allow_nan=False)
        stream.write("\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write a nonempty table using the first row's key order."""
    if not rows:
        raise ValueError("Cannot write an empty results table.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
