"""Classify HQ-WRI combinations using annual or pooled spatial-mean thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from workflow_utils import (
    assert_same_grid, ensure_new_directory, load_config, read_raster,
    resolve_path, write_csv, write_json, write_raster,
)


def classify_states(hq, wri, hq_threshold, wri_threshold):
    """Return 1=HH, 2=HL, 3=LH, 4=LL; equality belongs to the high class."""
    if hq.shape != wri.shape:
        raise ValueError("HQ and WRI dimensions differ.")
    if not np.isfinite([hq_threshold, wri_threshold]).all():
        raise ValueError("Both classification thresholds must be finite.")
    valid = np.isfinite(hq) & np.isfinite(wri)
    result = np.full(hq.shape, np.nan, dtype=np.float32)
    high_hq, high_wri = hq >= hq_threshold, wri >= wri_threshold
    result[valid & high_hq & high_wri] = 1
    result[valid & high_hq & ~high_wri] = 2
    result[valid & ~high_hq & high_wri] = 3
    result[valid & ~high_hq & ~high_wri] = 4
    return result


def annual_thresholds(hq, wri):
    """Use paired valid wetland pixels, as in the original annual-threshold script."""
    valid = np.isfinite(hq) & np.isfinite(wri)
    if not valid.any():
        raise ValueError("No jointly valid wetland pixels for classification.")
    return float(np.mean(hq[valid])), float(np.mean(wri[valid]))


def load_inputs(config, year, mask, mask_metadata):
    hq_path = resolve_path(config, config["paths"]["habitat_quality"].format(year=year))
    wri_path = resolve_path(config, config["paths"]["wri"].format(year=year))
    hq, metadata = read_raster(hq_path)
    wri, wri_metadata = read_raster(wri_path)
    assert_same_grid(metadata, wri_metadata, str(wri_path))
    if mask is not None:
        assert_same_grid(metadata, mask_metadata, "wetland mask")
        hq[mask != 1], wri[mask != 1] = np.nan, np.nan
    return hq, wri, metadata


def run(config: dict) -> None:
    mode = config["classification"]["threshold_mode"]
    if mode not in ("annual", "pooled"):
        raise ValueError("threshold_mode must be 'annual' or 'pooled'.")
    mask_path = config["paths"].get("wetland_mask")
    mask, mask_metadata = (None, None)
    if mask_path:
        mask, mask_metadata = read_raster(resolve_path(config, mask_path))
        if not np.any(mask == 1):
            raise ValueError("The wetland mask must contain wetland pixels coded as 1.")

    # The original pooled script uses each variable's own finite wetland pixels.
    # The original annual script uses their paired-valid intersection instead.
    pooled = None
    reference = None
    hq_values, wri_values = [], []
    for year in config["years"]:
        hq, wri, metadata = load_inputs(config, year, mask, mask_metadata)
        if reference is None:
            reference = metadata
        assert_same_grid(reference, metadata, str(year))
        annual_thresholds(hq, wri)  # Fail before writing if a year has no valid pairs.
        if mode == "pooled":
            hq_values.append(hq[np.isfinite(hq)])
            wri_values.append(wri[np.isfinite(wri)])
    if mode == "pooled":
        pooled = (float(np.mean(np.concatenate(hq_values))),
                  float(np.mean(np.concatenate(wri_values))))
    del hq_values, wri_values

    destinations = [
        resolve_path(config, config["paths"]["mismatch"].format(year=year))
        for year in config["years"]
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("The mismatch output pattern must distinguish every year.")
    if len({path.parent for path in destinations}) != 1:
        raise ValueError("All mismatch files must share one output directory.")
    output = ensure_new_directory(destinations[0].parent)
    rows = []
    for year, destination in zip(config["years"], destinations):
        hq, wri, metadata = load_inputs(config, year, mask, mask_metadata)
        thresholds = pooled if pooled is not None else annual_thresholds(hq, wri)
        states = classify_states(hq, wri, *thresholds)
        write_raster(destination, states, metadata)
        rows.append({
            "year": year, "threshold_mode": mode,
            "hq_threshold": thresholds[0], "wri_threshold": thresholds[1],
            "valid_pixels": int(np.isfinite(states).sum()),
            **{name: int(np.sum(states == code))
               for code, name in enumerate(("HH", "HL", "LH", "LL"), start=1)},
        })
        print(f"Saved mismatch classes: {destination.name}")
    write_csv(output / "thresholds_and_counts.csv", rows)
    write_json(output / "configuration.json", {
        k: v for k, v in config.items() if not k.startswith("_")
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
