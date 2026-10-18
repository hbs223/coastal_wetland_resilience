"""Synthetic regression tests; all generated data remain in temporary directories.

Run from the release directory with ``python -m unittest discover -s tests -v``.
The smoke test uses two tiny candidates (two and three trees), never the release
configuration's full 50-candidate search. No private research inputs are read.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import workflow_utils as utils


def import_stage(filename):
    specification = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


wri_stage = import_stage("01_compute_wetland_resilience.py")
classification = import_stage("02_classify_habitat_resilience.py")
preparation = import_stage("03_prepare_training_data.py")
training = import_stage("04_train_xgboost_random_search.py")


def grid(shape=(2, 3), epsg=3857, transform=(1000, 100, 0, 2000, 0, -100)):
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(epsg)
    return {"shape": shape, "transform": transform, "projection": reference.ExportToWkt()}


def configuration(base):
    with (ROOT / "config.json").open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    config["years"] = [2002, 2006]
    config["_base_dir"] = Path(base)
    return config


def save_input(base, pattern, year, values, metadata=None):
    values = np.asarray(values, dtype=np.float32)
    utils.write_raster(Path(base) / pattern.format(year=year), values, metadata or grid(values.shape))


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


class RasterAndOutputTests(unittest.TestCase):
    def test_nodata_georeferencing_and_float32_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.tif"
            values = np.array([[0, 1, np.nan], [2, np.inf, -3]], dtype=np.float32)
            metadata = grid(values.shape)
            utils.write_raster(path, values, metadata)
            recovered, observed = utils.read_raster(path)
            expected = values.copy()
            expected[~np.isfinite(expected)] = np.nan
            np.testing.assert_allclose(recovered, expected, equal_nan=True)
            self.assertEqual(recovered.dtype, np.dtype("float32"))
            utils.assert_same_grid(metadata, observed, "round trip")
            dataset = gdal.Open(str(path))
            self.assertEqual(dataset.GetRasterBand(1).GetNoDataValue(), -9999)
            self.assertEqual(dataset.RasterCount, 1)
            dataset = None
            with self.assertRaises(FileExistsError):
                utils.write_raster(path, expected, metadata)

    def test_declared_unsigned_nodata_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsigned.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(str(path), 3, 2, 1, gdal.GDT_UInt16)
            metadata = grid()
            dataset.SetGeoTransform(metadata["transform"])
            dataset.SetProjection(metadata["projection"])
            dataset.GetRasterBand(1).SetNoDataValue(65535)
            dataset.GetRasterBand(1).WriteArray(np.array([[0, 10000, 65535], [1, 2, 3]], dtype=np.uint16))
            dataset = None
            recovered, observed = utils.read_raster(path)
            self.assertTrue(np.isnan(recovered[0, 2]))
            self.assertEqual(recovered[0, 1], 10000)
            utils.assert_same_grid(metadata, observed, "unsigned input")

    def test_grid_shape_alignment_and_crs_mismatches_rejected(self):
        reference = grid()
        candidates = [grid(shape=(3, 2)), grid(transform=(1001, 100, 0, 2000, 0, -100)), grid(epsg=4326)]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                utils.assert_same_grid(reference, candidate, "misaligned input")
        utils.assert_same_grid(reference, copy.deepcopy(reference), "same grid")

    def test_multiband_and_missing_crs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, bands, projection in (("multiband.tif", 2, grid()["projection"]), ("no_crs.tif", 1, "")):
                path = Path(directory) / name
                dataset = gdal.GetDriverByName("GTiff").Create(str(path), 3, 2, bands, gdal.GDT_Float32)
                if projection:
                    dataset.SetProjection(projection)
                dataset = None
                with self.subTest(name=name), self.assertRaises(ValueError):
                    utils.read_raster(path)

    def test_output_directory_guard_allows_only_empty_or_gitkeep(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            utils.ensure_new_directory(destination)
            utils.ensure_new_directory(destination)
            (destination / ".gitkeep").touch()
            utils.ensure_new_directory(destination)
            sentinel = destination / "prior_result.json"
            sentinel.write_text('{"preserved":true}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                utils.ensure_new_directory(destination)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"preserved":true}')


class WetlandResilienceTests(unittest.TestCase):
    def test_reverse_pd_geometric_mean_zero_and_missing(self):
        inputs = {
            "patch_density": np.array([[0, 10, 5, np.nan]], dtype=np.float32),
            "cohesion": np.array([[4, 4, 3, 4]], dtype=np.float32),
            "vegetation_resilience": np.array([[8, 8, 4, 8]], dtype=np.float32),
            "hydrological_resilience": np.array([[2, 2, 1, 2]], dtype=np.float32),
        }
        extrema = {"patch_density": (0, 10), "cohesion": (0, 4), "vegetation_resilience": (0, 8), "hydrological_resilience": (0, 2)}
        result = wri_stage.calculate_wri(inputs, extrema)
        expected = np.array([[1, 0, (0.5 * 0.75 * 0.5 * 0.5) ** 0.25, np.nan]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-6)
        inputs["cohesion"][0, 0] = np.nan
        self.assertTrue(np.isnan(wri_stage.calculate_wri(inputs, extrema)[0, 0]))

    def test_constant_nonfinite_and_reversed_extrema_rejected(self):
        for low, high in ((1, 1), (2, 1), (np.nan, 2), (0, np.inf)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                wri_stage.normalize_indicator(np.array([1], dtype=np.float32), low, high)

    def test_pooled_extrema_across_years_not_annual(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            annual = {2002: np.array([[0, 1, 2]], dtype=np.float32), 2006: np.array([[2, 3, 4]], dtype=np.float32)}
            for year, values in annual.items():
                for name, pattern in config["indicators"].items():
                    save_input(directory, pattern, year, values)
            with contextlib.redirect_stdout(io.StringIO()):
                wri_stage.run(config)
            record = read_json(Path(directory) / "outputs/wri/normalization.json")
            for bounds in record["pooled_extrema"].values():
                self.assertEqual(bounds, [0.0, 4.0])
            for year, values in annual.items():
                scaled = values / 4
                expected = ((1 - scaled) * scaled ** 3) ** 0.25
                actual, _ = utils.read_raster(Path(directory) / config["paths"]["wri"].format(year=year))
                np.testing.assert_allclose(actual, expected, rtol=1e-6)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                wri_stage.run(config)

    def test_constant_inputs_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            for year in config["years"]:
                for pattern in config["indicators"].values():
                    save_input(directory, pattern, year, np.ones((2, 3)))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                wri_stage.run(config)
            self.assertFalse((Path(directory) / "outputs/wri").exists())


class ClassificationTests(unittest.TestCase):
    def test_annual_means_use_paired_valid_pixels(self):
        hq = np.array([[0, 2, 100, np.nan]], dtype=np.float32)
        wri = np.array([[2, 4, np.nan, 200]], dtype=np.float32)
        self.assertEqual(classification.annual_thresholds(hq, wri), (1.0, 3.0))
        with self.assertRaises(ValueError):
            classification.annual_thresholds(np.array([np.nan]), np.array([1]))

    def test_threshold_equality_is_high_and_class_mapping(self):
        hq = np.array([[0.5, 0.5, 0.4, 0.4, np.nan]], dtype=np.float32)
        wri = np.array([[0.5, 0.4, 0.5, 0.4, 0.9]], dtype=np.float32)
        result = classification.classify_states(hq, wri, 0.5, 0.5)
        np.testing.assert_allclose(result, [[1, 2, 3, 4, np.nan]], equal_nan=True)
        with self.assertRaises(ValueError):
            classification.classify_states(hq, wri, np.nan, 0.5)

    def test_pooled_uses_each_variables_own_finite_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            config["classification"]["threshold_mode"] = "pooled"
            config["paths"]["wetland_mask"] = None
            annual = {
                2002: (np.array([[0, 2, 100, np.nan]], dtype=np.float32), np.array([[2, 4, np.nan, 200]], dtype=np.float32)),
                2006: (np.array([[4, 6, np.nan, np.nan]], dtype=np.float32), np.array([[6, 8, np.nan, 100]], dtype=np.float32)),
            }
            for year, (hq, wri) in annual.items():
                save_input(directory, config["paths"]["habitat_quality"], year, hq)
                save_input(directory, config["paths"]["wri"], year, wri)
            with contextlib.redirect_stdout(io.StringIO()):
                classification.run(config)
            with (Path(directory) / "outputs/mismatch/thresholds_and_counts.csv").open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            expected_hq = np.mean([0, 2, 100, 4, 6])
            expected_wri = np.mean([2, 4, 200, 6, 8, 100])
            for record in records:
                self.assertAlmostEqual(float(record["hq_threshold"]), expected_hq, places=5)
                self.assertAlmostEqual(float(record["wri_threshold"]), expected_wri, places=5)
                self.assertEqual(int(record["valid_pixels"]), 2)
            for year, (hq, wri) in annual.items():
                actual, _ = utils.read_raster(Path(directory) / config["paths"]["mismatch"].format(year=year))
                expected = classification.classify_states(hq, wri, expected_hq, expected_wri)
                np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_annual_classification_applies_wetland_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            config["classification"]["threshold_mode"] = "annual"
            mask = np.array([[1, 1, 0, np.nan]], dtype=np.float32)
            save_input(directory, config["paths"]["wetland_mask"], 2002, mask)
            for year in config["years"]:
                hq = np.array([[0, 2, 100, 100]], dtype=np.float32)
                wri = np.array([[2, 4, 100, 100]], dtype=np.float32)
                save_input(directory, config["paths"]["habitat_quality"], year, hq)
                save_input(directory, config["paths"]["wri"], year, wri)
            with contextlib.redirect_stdout(io.StringIO()):
                classification.run(config)
            for year in config["years"]:
                actual, _ = utils.read_raster(Path(directory) / config["paths"]["mismatch"].format(year=year))
                np.testing.assert_allclose(actual, [[4, 1, np.nan, np.nan]], equal_nan=True)
            with (Path(directory) / "outputs/mismatch/thresholds_and_counts.csv").open(encoding="utf-8", newline="") as stream:
                for record in csv.DictReader(stream):
                    self.assertEqual(float(record["hq_threshold"]), 1)
                    self.assertEqual(float(record["wri_threshold"]), 3)


class SpatialModelTests(unittest.TestCase):
    def test_block_split_and_cv_are_disjoint_and_exhaustive(self):
        row, col = np.indices((20, 20))
        mismatch = np.where(col % 2 == 0, 2, 1).astype(np.float32)
        arrays, summary = preparation.prepare_arrays(mismatch, [row.astype(np.float32), col.astype(np.float32)], block_size=4)
        self.assertEqual(summary["valid_samples"], 400)
        self.assertEqual(summary["train_block_count"], 20)
        self.assertEqual(summary["test_block_count"], 5)
        self.assertFalse(np.intersect1d(arrays["train_block_id"], arrays["test_block_id"]).size)
        training.validate_training_arrays(arrays, 2)
        folds = training.spatial_folds(arrays["train_block_id"], arrays["y_train"])
        visits = np.zeros(len(arrays["y_train"]), dtype=int)
        for train_mask, validation_mask in folds:
            self.assertFalse(np.intersect1d(arrays["train_block_id"][train_mask], arrays["train_block_id"][validation_mask]).size)
            self.assertTrue(np.all(train_mask ^ validation_mask))
            visits += validation_mask
        np.testing.assert_array_equal(visits, np.ones_like(visits))
        broken = {name: values.copy() for name, values in arrays.items()}
        broken["test_block_id"][0] = broken["train_block_id"][0]
        with self.assertRaises(ValueError):
            training.validate_training_arrays(broken, 2)

    def test_probability_equality_binary_f1_and_invalid_labels(self):
        metrics, predicted = training.evaluation_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.5, 0.5, 0.9]))
        np.testing.assert_array_equal(predicted, [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["F1"], 0.8)
        self.assertEqual(metrics["F1_averaging"], "binary")
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        with self.assertRaises(ValueError):
            preparation.validate_mismatch_labels(np.array([[1, 2.5]]))

    def test_invalid_class_splits_and_excess_candidate_count_rejected(self):
        with self.assertRaises(ValueError):
            preparation.prepare_arrays(np.ones((20, 20)), [np.ones((20, 20))], block_size=4)
        with self.assertRaises(ValueError):
            preparation.split_by_blocks(np.array([0, 1, 2, 3, 4]), min_train_blocks=5)
        model = configuration(ROOT)["model"]
        model["param_grid"] = {name: values[:1] for name, values in model["param_grid"].items()}
        model["n_iter"] = 2
        with self.assertRaises(ValueError):
            training.parameter_candidates(model)

    def test_classification_provenance_rejects_stale_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            path = utils.resolve_path(
                config, config["paths"]["mismatch"].format(year=2002)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                external = preparation.classification_provenance(config, path)
            self.assertEqual(external["status"], "external_input_not_verified")
            utils.write_json(
                path.parent / "configuration.json",
                {key: value for key, value in config.items() if not key.startswith("_")},
            )
            verified = preparation.classification_provenance(config, path)
            self.assertEqual(verified["status"], "step_02_configuration_verified")
            changed = copy.deepcopy(config)
            changed["classification"]["threshold_mode"] = "annual"
            with self.assertRaises(ValueError):
                preparation.classification_provenance(changed, path)
            changed = copy.deepcopy(config)
            changed["paths"]["wetland_mask"] = "data/different_mask.tif"
            with self.assertRaises(ValueError):
                preparation.classification_provenance(changed, path)

    def test_tiny_full_pipeline_default_config_and_single_year_cli(self):
        """Two input years; one selected model year; 2 x 5 CV fits + one refit."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = configuration(base)
            config["sampling"]["block_size_pixels"] = 4
            config["model"].update({"n_iter": 2, "n_jobs": 1, "n_splits": 5})
            config["model"]["param_grid"] = {
                "n_estimators": [2, 3], "max_depth": [2], "learning_rate": [0.1],
                "subsample": [1.0], "colsample_bytree": [1.0],
                "min_child_weight": [1], "gamma": [0.0],
            }
            row, col = np.indices((20, 20))
            patterns = {
                "patch_density": np.take([0, 1, 0, 1], col % 4),
                "cohesion": np.take([1, 0, 0, 1], col % 4),
                "vegetation_resilience": np.take([1, 0, 0, 1], col % 4),
                "hydrological_resilience": np.take([1, 0, 0, 1], col % 4),
            }
            save_input(base, config["paths"]["wetland_mask"], 2002, np.ones((20, 20)))
            for year in config["years"]:
                for name, pattern in config["indicators"].items():
                    save_input(base, pattern, year, patterns[name])
                save_input(base, config["paths"]["habitat_quality"], year, np.take([1, 1, 0, 0], col % 4))
                for number, specification in enumerate(config["predictors"]):
                    # A deterministic local signal and spatial gradients in all ten inputs.
                    values = ((col % 4 == 1) * 20 + row / 20 + col / 200 + number).astype(np.float32)
                    save_input(base, specification["path"], year, values)
            utils.write_json(base / "config.json", {key: value for key, value in config.items() if not key.startswith("_")})
            # A copy beside config.json tests the advertised default-config path
            # from an unrelated working directory without touching release files.
            script_names = ["workflow_utils.py", "01_compute_wetland_resilience.py", "02_classify_habitat_resilience.py", "03_prepare_training_data.py", "04_train_xgboost_random_search.py"]
            for name in script_names:
                shutil.copy2(ROOT / name, base / name)
            for name in script_names[1:]:
                arguments = [sys.executable, str(base / name)]
                if name.startswith(("03", "04")):
                    arguments += ["--years", "2006"]
                result = subprocess.run(arguments, cwd=base.parent, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, msg=f"{name}\n{result.stdout}\n{result.stderr}")
            self.assertFalse((base / "outputs/training_data/2002").exists())
            self.assertFalse((base / "outputs/models/2002").exists())
            split_directory = base / "outputs/training_data/2006"
            summary = read_json(split_directory / "split_summary.json")
            self.assertEqual(summary["feature_count"], 10)
            self.assertEqual(summary["train_test_block_overlap"], 0)
            self.assertEqual(summary["train_samples"] + summary["test_samples"], 400)
            train_blocks = np.load(split_directory / "train_block_id.npy", allow_pickle=False)
            test_blocks = np.load(split_directory / "test_block_id.npy", allow_pickle=False)
            self.assertFalse(np.intersect1d(train_blocks, test_blocks).size)
            model_directory = base / "outputs/models/2006"
            self.assertIn("learner", read_json(model_directory / "model.json"))
            metadata = read_json(model_directory / "run_metadata.json")
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(len(metadata["candidate_list"]), 2)
            self.assertEqual(metadata["importance_type"], "gain")
            self.assertEqual(len(metadata["cross_validation_folds"]), 5)
            self.assertIn("xgboost", metadata["versions"])
            self.assertFalse(metadata["early_stopping"])
            for fold in metadata["cross_validation_folds"]:
                self.assertFalse(set(fold["training_block_ids"]) & set(fold["validation_block_ids"]))
            metrics = read_json(model_directory / "test_metrics.json")
            self.assertEqual(metrics["F1_averaging"], "binary")
            self.assertTrue(0 <= metrics["ROC_AUC"] <= 1)
            with (model_directory / "CV_fold_results.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            with (model_directory / "feature_importance.csv").open(encoding="utf-8", newline="") as stream:
                importance = list(csv.DictReader(stream))
            self.assertEqual(len(importance), 10)
            total = sum(float(record["Importance"]) for record in importance)
            self.assertTrue(np.isclose(total, 0) or np.isclose(total, 1))
            # Native JSON must be loadable, not merely have a .json extension.
            from xgboost import XGBClassifier
            restored = XGBClassifier()
            restored.load_model(model_directory / "model.json")
            probabilities = restored.predict_proba(np.load(split_directory / "X_test.npy", allow_pickle=False))[:, 1]
            self.assertEqual(len(probabilities), summary["test_samples"])
            self.assertTrue(np.isfinite(probabilities).all())
            # A changed method must not silently reuse a split prepared under
            # different thresholds, predictor order/scales, or block geometry.
            for key, value in (
                ("classification", {"threshold_mode": "annual"}),
                ("sampling", {**config["sampling"], "block_size_pixels": 8}),
                ("years", [2006]),
            ):
                changed = copy.deepcopy(config)
                changed[key] = value
                with self.subTest(changed_key=key), self.assertRaises(ValueError):
                    training.load_year_data(changed, 2006)


if __name__ == "__main__":
    unittest.main()
