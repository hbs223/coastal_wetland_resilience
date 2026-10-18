"""Normalize four indicators across assessment years and calculate WRI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from workflow_utils import (
    assert_same_grid, ensure_new_directory, load_config, read_raster,
    resolve_path, write_json, write_raster,
)

INDICATORS = (
    "patch_density", "cohesion", "vegetation_resilience", "hydrological_resilience"
)


def normalize_indicator(values, minimum, maximum, reverse=False):
    """Apply pooled min-max scaling; reverse patch density only."""
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError("Normalization requires finite, distinct pooled extrema.")
    scaled = (maximum - values if reverse else values - minimum) / (maximum - minimum)
    return np.clip(scaled, 0, 1)


def calculate_wri(indicators: dict, extrema: dict) -> np.ndarray:
    """Compute the fourth root of the product, requiring four valid components."""
    normalized = [
        normalize_indicator(
            indicators[name], *extrema[name], reverse=(name == "patch_density")
        )
        for name in INDICATORS
    ]
    valid = np.logical_and.reduce([np.isfinite(values) for values in normalized])
    result = np.full(normalized[0].shape, np.nan, dtype=np.float32)
    product = np.ones(int(valid.sum()), dtype=np.float32)
    for values in normalized:
        product *= values[valid]
    result[valid] = np.power(product, 0.25)
    return result


def run(config: dict) -> None:
    """Scan pooled extrema, then process one year at a time to limit memory use."""
    extrema = {name: [np.inf, -np.inf] for name in INDICATORS}
    reference = None
    for year in config["years"]:
        for name in INDICATORS:
            path = resolve_path(config, config["indicators"][name].format(year=year))
            values, metadata = read_raster(path)
            if reference is None:
                reference = metadata
            assert_same_grid(reference, metadata, str(path))
            valid = values[np.isfinite(values)]
            if valid.size:
                extrema[name][0] = min(extrema[name][0], float(valid.min()))
                extrema[name][1] = max(extrema[name][1], float(valid.max()))
        print(f"Scanned indicator extrema: {year}")
    for name, (minimum, maximum) in extrema.items():
        if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
            raise ValueError(f"No finite, nonconstant pooled indicator range: {name}")

    destinations = [
        resolve_path(config, config["paths"]["wri"].format(year=year))
        for year in config["years"]
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("The WRI output pattern must distinguish every year.")
    if len({path.parent for path in destinations}) != 1:
        raise ValueError("All WRI files must share one output directory.")
    output = ensure_new_directory(destinations[0].parent)
    write_json(output / "normalization.json", {
        "years": config["years"], "pooled_extrema": extrema,
        "formula": "(PD_reverse * COHESION * VFR * HR) ** 0.25",
        "configuration": {k: v for k, v in config.items() if not k.startswith("_")},
    })
    for year, destination in zip(config["years"], destinations):
        indicators = {}
        for name in INDICATORS:
            path = resolve_path(config, config["indicators"][name].format(year=year))
            indicators[name], metadata = read_raster(path)
            assert_same_grid(reference, metadata, str(path))
        write_raster(destination, calculate_wri(indicators, extrema), reference)
        print(f"Saved WRI: {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
