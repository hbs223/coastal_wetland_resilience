"""Prepare annual binary-classification arrays using spatially disjoint blocks.

The original workflow's class mapping, predictor scaling, row-major block
numbering, and seeded 80/20 block split are retained. Missing raster observations
are excluded by a joint validity mask; invalid labels or misaligned grids fail
explicitly. This script does not resample rasters or rebalance classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workflow_utils import (
    assert_same_grid,
    ensure_new_directory,
    load_config,
    read_raster,
    resolve_path,
    write_json,
)


def create_blocks(rows: int, cols: int, block_size: int = 20) -> np.ndarray:
    """Number square pixel blocks in exactly the original row-major order."""
    if any(isinstance(value, bool) or int(value) != value or value <= 0
           for value in (rows, cols, block_size)):
        raise ValueError("Raster dimensions and block_size must be positive integers.")
    rows, cols, block_size = int(rows), int(cols), int(block_size)
    block_columns = (cols + block_size - 1) // block_size
    block_rows = (rows + block_size - 1) // block_size
    if block_columns * block_rows > np.iinfo(np.int32).max:
        raise ValueError("The block grid exceeds the supported int32 identifier range.")
    return (
        (np.arange(rows, dtype=np.int64) // block_size)[:, None] * block_columns
        + (np.arange(cols, dtype=np.int64) // block_size)[None, :]
    ).astype(np.int32)


def validate_mismatch_labels(mismatch: np.ndarray) -> None:
    """Allow NaN NoData and only the four finite integer labels 1, 2, 3, 4."""
    if mismatch.ndim != 2 or min(mismatch.shape) == 0:
        raise ValueError("The mismatch raster must be a nonempty two-dimensional array.")
    if np.isinf(mismatch).any():
        raise ValueError("The mismatch raster contains infinite values.")
    finite = mismatch[np.isfinite(mismatch)]
    unexpected = np.unique(finite[~np.isin(finite, [1, 2, 3, 4])])
    if unexpected.size:
        raise ValueError(f"Unexpected mismatch labels: {unexpected.tolist()}; expected 1–4.")


def require_both_classes(labels: np.ndarray, label: str) -> None:
    """Require a nonempty binary response containing both classes."""
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"{label} must contain both binary classes 0 and 1.")


def split_by_blocks(
    block_ids: np.ndarray,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    min_train_blocks: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sample masks and block lists without retrying an unsuitable split.

    RandomState.shuffle reproduces np.random.seed(seed); np.random.shuffle
    from the original script, without changing NumPy's global random state.
    """
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    if block_ids.ndim != 1 or not block_ids.size:
        raise ValueError("block_ids must be a nonempty one-dimensional array.")
    if (not np.isfinite(block_ids).all() or np.any(block_ids < 0)
            or np.any(block_ids != np.floor(block_ids))):
        raise ValueError("Block identifiers must be finite nonnegative integers.")
    if int(min_train_blocks) != min_train_blocks or min_train_blocks < 5:
        raise ValueError("At least five training blocks are required.")
    blocks = np.unique(block_ids)
    np.random.RandomState(random_seed).shuffle(blocks)
    split_index = int(len(blocks) * train_fraction)
    train_blocks, test_blocks = blocks[:split_index], blocks[split_index:]
    if len(train_blocks) < min_train_blocks:
        raise ValueError(
            f"Only {len(train_blocks)} training blocks; at least {min_train_blocks} are required."
        )
    if not len(test_blocks):
        raise ValueError("The configured split leaves no test blocks.")
    train_mask = np.isin(block_ids, train_blocks)
    test_mask = np.isin(block_ids, test_blocks)
    if np.intersect1d(train_blocks, test_blocks).size or not np.all(train_mask ^ test_mask):
        raise ValueError("Training and test blocks must form a disjoint, exhaustive split.")
    return train_mask, test_mask, train_blocks, test_blocks


def prepare_arrays(
    mismatch: np.ndarray,
    feature_arrays: list[np.ndarray],
    block_size: int = 20,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    min_train_blocks: int = 5,
) -> tuple[dict[str, np.ndarray], dict]:
    """Create the eight original arrays and a machine-readable split summary."""
    validate_mismatch_labels(mismatch)
    if not feature_arrays:
        raise ValueError("At least one predictor is required.")
    for index, array in enumerate(feature_arrays):
        if array.shape != mismatch.shape or array.ndim != 2:
            raise ValueError(f"Predictor {index} has a different shape from the mismatch raster.")
        if np.isinf(array).any():
            raise ValueError(f"Predictor {index} contains infinite values.")
    predictors = np.stack(feature_arrays, axis=-1).astype(np.float32)
    if np.isinf(predictors).any():
        raise ValueError("Converting predictors to float32 produced infinite values.")
    valid = np.isfinite(mismatch) & np.all(np.isfinite(predictors), axis=-1)
    if not valid.any():
        raise ValueError("No pixels have valid mismatch labels and all predictors.")
    original_labels = mismatch[valid].astype(np.int32)
    binary_labels = (original_labels == 2).astype(np.int32)
    require_both_classes(binary_labels, "All valid pixels")
    valid_predictors = predictors[valid]
    block_ids = create_blocks(*mismatch.shape, block_size)[valid]
    train, test, train_blocks, test_blocks = split_by_blocks(
        block_ids, train_fraction, random_seed, min_train_blocks
    )
    require_both_classes(binary_labels[train], "Training split")
    require_both_classes(binary_labels[test], "Test split")
    arrays = {
        "X_train": valid_predictors[train],
        "X_test": valid_predictors[test],
        "y_train": binary_labels[train],
        "y_test": binary_labels[test],
        "y_original_train": original_labels[train],
        "y_original_test": original_labels[test],
        "train_block_id": block_ids[train],
        "test_block_id": block_ids[test],
    }

    def distribution(values: np.ndarray) -> dict[str, int]:
        labels, counts = np.unique(values, return_counts=True)
        return {str(int(label)): int(count) for label, count in zip(labels, counts)}

    summary = {
        "raster_shape": list(mismatch.shape),
        "total_raster_pixels": int(mismatch.size),
        "valid_samples": int(valid.sum()),
        "excluded_missing_samples": int(mismatch.size - valid.sum()),
        "feature_count": len(feature_arrays),
        "block_size_pixels": int(block_size),
        "block_numbering": "row-major, top-left origin, partial edge blocks retained",
        "random_generator": "numpy.random.RandomState",
        "random_seed": int(random_seed),
        "requested_train_block_fraction": float(train_fraction),
        "actual_train_sample_fraction": float(train.sum() / valid.sum()),
        "all_valid_blocks": int(len(np.unique(block_ids))),
        "train_block_count": int(len(train_blocks)),
        "test_block_count": int(len(test_blocks)),
        "train_blocks_in_shuffled_order": train_blocks.tolist(),
        "test_blocks_in_shuffled_order": test_blocks.tolist(),
        "train_samples": int(train.sum()),
        "test_samples": int(test.sum()),
        "binary_class_mapping": {"1": 0, "2": 1, "3": 0, "4": 0},
        "all_binary_class_counts": distribution(binary_labels),
        "train_binary_class_counts": distribution(binary_labels[train]),
        "test_binary_class_counts": distribution(binary_labels[test]),
        "all_original_class_counts": distribution(original_labels),
        "train_original_class_counts": distribution(original_labels[train]),
        "test_original_class_counts": distribution(original_labels[test]),
        "train_test_block_overlap": 0,
    }
    return arrays, summary


def selected_years(config: dict, requested: list[int] | None) -> list[int]:
    """Validate requested years against the configured analysis years."""
    years = config["years"] if requested is None else requested
    if not years or len(set(years)) != len(years):
        raise ValueError("Years must be a nonempty list without duplicates.")
    if any(isinstance(year, bool) or int(year) != year for year in years):
        raise ValueError("All years must be integers.")
    if any(year not in config["years"] for year in years):
        raise ValueError("--years must be a subset of the configured years.")
    return [int(year) for year in years]


def classification_provenance(config: dict, mismatch_path: Path) -> dict:
    """Check step-02 metadata so stale labels cannot acquire a new method label."""
    sidecar = mismatch_path.parent / "configuration.json"
    if not sidecar.is_file():
        print(
            "Warning: external mismatch raster has no step-02 configuration: "
            f"{mismatch_path}. Its thresholds must be verified separately."
        )
        return {
            "status": "external_input_not_verified",
            "note": "The current classification setting was not applied to this input by step 03.",
        }
    with sidecar.open(encoding="utf-8-sig") as stream:
        source_config = json.load(stream)
    for section in ("years", "classification"):
        if source_config.get(section) != config.get(section):
            raise ValueError(
                f"Mismatch-raster {section} settings differ from the current configuration. "
                "Rerun classification in a new output directory or restore its original settings."
            )
    for name in ("habitat_quality", "wri", "wetland_mask", "mismatch"):
        if source_config.get("paths", {}).get(name) != config["paths"].get(name):
            raise ValueError(f"Mismatch-raster source path settings differ: {name}.")
    return {
        "status": "step_02_configuration_verified",
        "metadata_file": str(sidecar),
        "threshold_mode": source_config["classification"]["threshold_mode"],
    }


def prepare_year(config: dict, year: int) -> dict:
    """Read aligned rasters and save one annual split in a new output directory."""
    specifications = config["predictors"]
    names = [item["name"] for item in specifications]
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every predictor must have a nonempty name.")
    if len(set(names)) != len(names):
        raise ValueError("Predictor names must be unique.")
    mismatch_path = resolve_path(config, config["paths"]["mismatch"].format(year=year))
    provenance = classification_provenance(config, mismatch_path)
    mismatch, reference = read_raster(mismatch_path)
    feature_arrays, source_records = [], []
    for item in specifications:
        path = resolve_path(config, item["path"].format(year=year))
        array, metadata = read_raster(path)
        assert_same_grid(reference, metadata, item["name"])
        # Scaling is explicit in the configuration, including 0.1 for the
        # stored TerraClimate AET and climatic-water-deficit inputs.
        scale = float(item["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Predictor {item['name']} requires a finite, positive scale.")
        with np.errstate(over="raise", invalid="raise"):
            array = array * scale
        feature_arrays.append(array)
        source_records.append({"name": item["name"], "path": str(path), "scale": scale})
    sampling = config["sampling"]
    n_splits = config["model"]["n_splits"]
    if isinstance(n_splits, bool) or int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    arrays, summary = prepare_arrays(
        mismatch,
        feature_arrays,
        block_size=sampling["block_size_pixels"],
        train_fraction=sampling["train_fraction"],
        random_seed=sampling["random_seed"],
        min_train_blocks=max(5, int(n_splits)),
    )
    summary.update({
        "year": year,
        "mismatch_source": str(mismatch_path),
        "classification_provenance": provenance,
        "predictor_sources": source_records,
        "grid": reference,
        "config": {key: value for key, value in config.items() if key != "_base_dir"},
        "config_base_directory": str(config["_base_dir"]),
    })
    output = resolve_path(config, config["paths"]["training_data"]) / str(year)
    ensure_new_directory(output)
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    write_json(output / "feature_names.json", names)
    write_json(output / "split_summary.json", summary)
    print(
        f"{year}: saved {summary['train_samples']} training and "
        f"{summary['test_samples']} test samples to {output}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--years", type=int, nargs="+", help="Optional subset of configured years.")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    for year in selected_years(config, arguments.years):
        prepare_year(config, year)


if __name__ == "__main__":
    main()
