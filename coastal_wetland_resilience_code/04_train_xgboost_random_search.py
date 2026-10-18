"""Fit annual XGBoost models with randomized search and spatial-block CV.

The same seeded candidate list is reused for every year. Annual training-class
weights remain fixed across folds, and no early stopping or alternate split is
introduced. Models use XGBoost's native JSON format rather than Python pickle;
load a model with XGBClassifier().load_model(path).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, ParameterSampler

from workflow_utils import ensure_new_directory, load_config, resolve_path, write_csv, write_json


ARRAY_NAMES = (
    "X_train", "X_test", "y_train", "y_test", "y_original_train",
    "y_original_test", "train_block_id", "test_block_id",
)
SEARCH_PARAMETERS = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma",
}


def require_both_classes(labels: np.ndarray, label: str) -> None:
    """Reject invalid or one-class samples instead of altering the split."""
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"{label} must contain both binary classes 0 and 1.")


def validate_training_arrays(arrays: dict[str, np.ndarray], feature_count: int, n_splits: int = 5) -> None:
    """Check shapes, finite values, original labels, and block independence."""
    if isinstance(n_splits, bool) or int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if isinstance(feature_count, bool) or int(feature_count) != feature_count or feature_count < 1:
        raise ValueError("feature_count must be a positive integer.")
    for split in ("train", "test"):
        features = arrays[f"X_{split}"]
        labels = arrays[f"y_{split}"]
        original = arrays[f"y_original_{split}"]
        blocks = arrays[f"{split}_block_id"]
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != feature_count:
            raise ValueError(f"X_{split} has an invalid shape or feature count.")
        if not np.isfinite(features).all():
            raise ValueError(f"X_{split} contains missing or infinite values.")
        for name, values in (
            ("binary labels", labels), ("original labels", original),
            ("block identifiers", blocks),
        ):
            if values.ndim != 1 or len(values) != len(features) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} shape or nonfinite values in {split}.")
        require_both_classes(labels, f"{split} labels")
        if not np.isin(original, [1, 2, 3, 4]).all():
            raise ValueError(f"{split} original labels must belong to 1, 2, 3, 4.")
        if not np.array_equal(labels, (original == 2).astype(np.int32)):
            raise ValueError(f"{split} binary labels do not match the original class-2 mapping.")
        if np.any(blocks < 0) or np.any(blocks != np.floor(blocks)):
            raise ValueError(f"{split} block identifiers must be nonnegative integers.")
    train_blocks = np.unique(arrays["train_block_id"])
    test_blocks = np.unique(arrays["test_block_id"])
    if len(train_blocks) < max(5, n_splits):
        raise ValueError(f"At least {max(5, n_splits)} distinct training blocks are required.")
    if np.intersect1d(train_blocks, test_blocks).size:
        raise ValueError("Training and test sets share spatial block identifiers.")


def spatial_folds(
    block_ids: np.ndarray, labels: np.ndarray,
    n_splits: int = 5, random_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use shuffled KFold over unique block IDs, not over individual samples."""
    if block_ids.ndim != 1 or labels.ndim != 1 or len(block_ids) != len(labels):
        raise ValueError("Block identifiers and labels must be aligned one-dimensional arrays.")
    if (not np.isfinite(block_ids).all() or np.any(block_ids < 0)
            or np.any(block_ids != np.floor(block_ids))):
        raise ValueError("Spatial block identifiers must be finite nonnegative integers.")
    require_both_classes(labels, "Cross-validation input")
    if isinstance(n_splits, bool) or int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    blocks = np.unique(block_ids)
    if len(blocks) < max(5, n_splits):
        raise ValueError(f"At least {max(5, n_splits)} training blocks are required for cross-validation.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []
    validation_visits = np.zeros(len(labels), dtype=np.int32)
    for number, (training, validation) in enumerate(splitter.split(blocks), start=1):
        train_mask = np.isin(block_ids, blocks[training])
        validation_mask = np.isin(block_ids, blocks[validation])
        if np.any(train_mask & validation_mask) or not np.all(train_mask | validation_mask):
            raise ValueError(f"Fold {number} does not partition the training samples.")
        require_both_classes(labels[train_mask], f"Fold {number} training samples")
        require_both_classes(labels[validation_mask], f"Fold {number} validation samples")
        validation_visits += validation_mask
        folds.append((train_mask, validation_mask))
    if not np.all(validation_visits == 1):
        raise ValueError("Every training sample must be validated exactly once per candidate.")
    return folds


def parameter_candidates(model_config: dict) -> list[dict]:
    """Sample the search space once, rejecting silent candidate-count reductions."""
    grid = model_config["param_grid"]
    if set(grid) != SEARCH_PARAMETERS:
        raise ValueError(f"param_grid must contain exactly {sorted(SEARCH_PARAMETERS)}.")
    total = 1
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} requires a nonempty candidate list.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) for value in values):
            raise ValueError(f"{name} candidate values must be finite numbers.")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicate candidates.")
        if name in {"n_estimators", "max_depth"} and any(
            int(value) != value or value < 1 for value in values
        ):
            raise ValueError(f"{name} requires positive integer candidates.")
        if name in {"learning_rate", "subsample", "colsample_bytree"} and any(
            not 0 < value <= 1 for value in values
        ):
            raise ValueError(f"{name} candidates must lie in (0, 1].")
        if name in {"min_child_weight", "gamma"} and any(value < 0 for value in values):
            raise ValueError(f"{name} candidates must be nonnegative.")
        total *= len(values)
    count = model_config["n_iter"]
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= total:
        raise ValueError(f"n_iter must be an integer from 1 to {total}; no automatic reduction is applied.")
    return [dict(parameters) for parameters in ParameterSampler(
        grid, n_iter=int(count), random_state=model_config["random_seed"]
    )]


def make_classifier(parameters: dict, positive_weight: float, model_config: dict):
    """Use the original explicit parameters; record all other library defaults."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=positive_weight,
        random_state=model_config["random_seed"],
        n_jobs=model_config["n_jobs"],
        **parameters,
    )


def validate_probabilities(probabilities: np.ndarray, expected_count: int) -> None:
    """Reject nonfinite, out-of-range, or incorrectly shaped probabilities."""
    if (probabilities.shape != (expected_count,) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ValueError("Classifier returned invalid positive-class probabilities.")


def evaluation_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
    """Evaluate positive-class binary metrics with the original >= decision rule."""
    require_both_classes(labels, "Test labels")
    validate_probabilities(probabilities, len(labels))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("The probability threshold must lie strictly between zero and one.")
    predicted = (probabilities >= threshold).astype(np.int32)
    results = {
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "Kappa": float(cohen_kappa_score(labels, predicted)),
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "probability_threshold": float(threshold),
        "positive_class": 1,
        "F1_averaging": "binary",
        "confusion_matrix_label_order": [0, 1],
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    return results, predicted


def runtime_versions() -> dict[str, str]:
    """Record dependency versions because unspecified XGBoost defaults can vary."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "xgboost", "GDAL"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "distribution metadata unavailable"
    return versions


def load_year_data(config: dict, year: int) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """Read the prepared arrays and verify the configured feature order."""
    directory = resolve_path(config, config["paths"]["training_data"]) / str(year)
    arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False) for name in ARRAY_NAMES}
    with (directory / "feature_names.json").open(encoding="utf-8") as stream:
        names = json.load(stream)
    expected = [item["name"] for item in config["predictors"]]
    if not names or names != expected or len(set(names)) != len(names):
        raise ValueError("Prepared feature names/order do not match the configuration.")
    with (directory / "split_summary.json").open(encoding="utf-8") as stream:
        split_summary = json.load(stream)
    if split_summary.get("year") != year:
        raise ValueError("The preparation summary does not match the requested year.")
    prepared_config = split_summary.get("config", {})
    for section in ("years", "predictors", "sampling", "classification"):
        if prepared_config.get(section) != config.get(section):
            raise ValueError(
                f"Prepared-data {section} settings differ from this configuration. "
                "Regenerate training data in a new directory, or use its original settings."
            )
    validate_training_arrays(arrays, len(names), config["model"]["n_splits"])
    return arrays, names, split_summary


def train_year(config: dict, year: int, candidates: list[dict]) -> dict:
    """Tune, refit, evaluate, and save one annual model in a new directory."""
    arrays, names, split_summary = load_year_data(config, year)
    model_config = config["model"]
    threshold = float(model_config["probability_threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("probability_threshold must lie strictly between zero and one.")
    if not candidates:
        raise ValueError("The parameter candidate list is empty.")
    features, labels, block_ids = arrays["X_train"], arrays["y_train"], arrays["train_block_id"]
    folds = spatial_folds(block_ids, labels, model_config["n_splits"], model_config["random_seed"])
    positive_weight = float(np.count_nonzero(labels == 0) / np.count_nonzero(labels == 1))
    output = resolve_path(config, config["paths"]["models"]) / str(year)
    ensure_new_directory(output)
    run_metadata = {
        "year": year,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "versions": runtime_versions(),
        "config": {key: value for key, value in config.items() if key != "_base_dir"},
        "config_base_directory": str(config["_base_dir"]),
        "preparation_summary": split_summary,
        "feature_names": names,
        "candidate_list": candidates,
        "scale_pos_weight": positive_weight,
        "class_weight_scope": "fixed from the full annual training set for every CV fit and final refit",
        "early_stopping": False,
        "model_format": "native XGBoost JSON, not Python pickle",
        "parameter_tie_rule": "first sampled candidate attaining the maximum mean validation ROC-AUC",
    }
    write_json(output / "run_metadata.json", run_metadata)
    try:
        search_results, fold_results = [], []
        best_mean_auc, best_parameters, best_candidate_id = -np.inf, None, None
        for candidate_id, parameters in enumerate(candidates, start=1):
            scores = []
            for fold_number, (training, validation) in enumerate(folds, start=1):
                classifier = make_classifier(parameters, positive_weight, model_config)
                classifier.fit(features[training], labels[training])
                probabilities = classifier.predict_proba(features[validation])[:, 1]
                validate_probabilities(probabilities, int(validation.sum()))
                auc = float(roc_auc_score(labels[validation], probabilities))
                scores.append(auc)
                fold_results.append({
                    "candidate_id": candidate_id,
                    "fold": fold_number,
                    "ROC_AUC": auc,
                    "training_samples": int(training.sum()),
                    "validation_samples": int(validation.sum()),
                    "training_blocks": int(len(np.unique(block_ids[training]))),
                    "validation_blocks": int(len(np.unique(block_ids[validation]))),
                })
            mean_auc = float(np.mean(scores))
            search_results.append({"candidate_id": candidate_id, "mean_auc": mean_auc, **parameters})
            if mean_auc > best_mean_auc:
                best_mean_auc, best_parameters, best_candidate_id = mean_auc, dict(parameters), candidate_id
            print(
                f"{year}: candidate {candidate_id}/{len(candidates)}, "
                f"mean spatial CV ROC-AUC={mean_auc:.6f}"
            )
        if best_parameters is None:
            raise RuntimeError("No candidate produced a selected model.")
        final_model = make_classifier(best_parameters, positive_weight, model_config)
        final_model.fit(features, labels)
        probabilities = final_model.predict_proba(arrays["X_test"])[:, 1]
        metrics, predictions = evaluation_metrics(arrays["y_test"], probabilities, threshold)

        # Default tree-booster feature_importances_ is normalized mean gain.
        # Query gain explicitly to keep its meaning stable and include unused features.
        raw_gain = final_model.get_booster().get_score(importance_type="gain")
        gain = np.array([raw_gain.get(f"f{index}", 0.0) for index in range(len(names))], dtype=float)
        if not np.isfinite(gain).all() or np.any(gain < 0):
            raise ValueError("The fitted model returned invalid gain importance.")
        gain_sum = float(gain.sum())
        importance = gain / gain_sum if gain_sum > 0 else np.zeros_like(gain)
        importance_status = (
            "normalized to sum to one" if gain_sum > 0
            else "all zero: no positive-gain splits"
        )
        final_model.save_model(str(output / "model.json"))
        write_csv(
            output / "All_parameter_results.csv",
            sorted(search_results, key=lambda row: -row["mean_auc"]),
        )
        write_csv(output / "CV_fold_results.csv", fold_results)
        write_json(output / "best_parameters.json", {
            "candidate_id": best_candidate_id,
            "parameters": best_parameters,
            "mean_spatial_CV_ROC_AUC": best_mean_auc,
            "scale_pos_weight": positive_weight,
        })
        write_json(output / "test_metrics.json", metrics)
        write_csv(output / "test_predictions.csv", [
            {"sample_index": index, "block_id": int(block), "original_class": int(original),
             "observed_class": int(observed),
             "predicted_probability": float(probability),
             "predicted_class": int(predicted)}
            for index, (block, original, observed, probability, predicted) in enumerate(zip(
                arrays["test_block_id"], arrays["y_original_test"],
                arrays["y_test"], probabilities, predictions
            ))
        ])
        write_csv(output / "feature_importance.csv", sorted([
            {"Feature": name, "Importance": float(value), "Mean_gain": float(raw)}
            for name, value, raw in zip(names, importance, gain)
        ], key=lambda row: -row["Importance"]))
        run_metadata.update({
            "status": "completed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "best_candidate_id": best_candidate_id,
            "best_parameters": best_parameters,
            "best_mean_spatial_CV_ROC_AUC": best_mean_auc,
            "importance_type": "gain",
            "importance_normalization": importance_status,
            "resolved_booster_configuration": json.loads(final_model.get_booster().save_config()),
            "cross_validation_folds": [
                {"fold": number,
                 "training_block_ids": np.unique(block_ids[training]).tolist(),
                 "validation_block_ids": np.unique(block_ids[validation]).tolist()}
                for number, (training, validation) in enumerate(folds, start=1)
            ],
        })
        write_json(output / "run_metadata.json", run_metadata)
        print(f"{year}: saved model and evaluation outputs to {output}")
        return metrics
    except Exception as error:
        run_metadata.update({"status": "failed", "failure": f"{type(error).__name__}: {error}"})
        write_json(output / "run_metadata.json", run_metadata)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--years", type=int, nargs="+", help="Optional subset of configured years.")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    years = config["years"] if arguments.years is None else arguments.years
    if not years or len(set(years)) != len(years):
        raise ValueError("Years must be a nonempty list without duplicates.")
    if any(isinstance(year, bool) or int(year) != year or year not in config["years"] for year in years):
        raise ValueError("--years must contain integer years from the configuration.")
    candidates = parameter_candidates(config["model"])
    for year in years:
        train_year(config, int(year), candidates)


if __name__ == "__main__":
    main()
