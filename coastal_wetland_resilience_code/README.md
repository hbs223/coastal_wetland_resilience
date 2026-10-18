# Coastal wetland habitat quality and resilience

Python code for combining wetland resilience indicators, classifying habitat
quality (HQ) and Wetland Resilience Index (WRI) states, preparing annual spatial
training samples, and fitting XGBoost models by randomized parameter search.

Assessment years: **2002, 2006, 2010, 2014, 2018, and 2022**.

## Scope

This repository begins with **aligned, precomputed raster inputs**. The supplied
source folder contained these downstream analysis steps, not the upstream GEE
downloads, landscape-metric calculation, vegetation/hydrological component
calculation, raster alignment, or InVEST HQ runs. No input data, derived rasters,
training arrays, fitted models, or historical evaluation results are bundled.

The author-confirmed study settings are **pooled HQ-WRI thresholds across all
six assessment years and an approximately 80:20 split of occupied spatial
blocks**. These settings are selected in `config.json` and preserve the supplied
annual machine-learning workflow. The annual-threshold option is retained only
for separate comparisons; it is not the selected study configuration. See
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the precise definitions.

## Files

| File | Purpose |
| --- | --- |
| `config.json` | Input patterns, output paths, years, scaling, classification, and model settings |
| `01_compute_wetland_resilience.py` | Pooled normalization and four-component WRI |
| `02_classify_habitat_resilience.py` | Annual or pooled HQ-WRI classification |
| `03_prepare_training_data.py` | Annual binary labels and spatial training/test partitions |
| `04_train_xgboost_random_search.py` | Randomized parameter search, spatial CV, and held-out evaluation |
| `workflow_utils.py` | Only the shared path, raster, and serialization functions used here |
| `tests/test_workflow.py` | Numerical checks and small synthetic-data workflow tests |
| `environment.yml` | Conda environment specification |

`data/inputs/` and `outputs/` contain only zero-byte `.gitkeep` placeholders.
Git does not preserve completely empty directories. The placeholders carry no
data and can be left in place. Generated data and model outputs are excluded by
`.gitignore`.

## Installation

Use a separate Conda environment, particularly on Windows where GDAL requires
matching native libraries:

```shell
conda env create -f environment.yml
conda activate coastal-wetland-resilience
```

This environment file specifies dependencies, not a historical lockfile.
The software versions used for the cleanup tests are listed in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md). Each model run records its
actual versions and resolved XGBoost configuration.

## Configure inputs

Edit `config.json`. All relative paths are resolved against the directory
containing that configuration, **not** the current terminal directory. Absolute
paths are also accepted. Use `{year}` where a file changes by assessment year.
Input files may retain their existing names by changing the path patterns; it
is not necessary to move, overwrite, or rename the original research data.

All rasters must be single-band GeoTIFFs with the same dimensions, coordinate
reference system, pixel size, and pixel alignment. Declare missing values in
their NoData metadata. Missing or nonfinite observations are excluded, not
imputed. This repository does not reproject or resample inputs.

| Input | Default file pattern | Requirement |
| --- | --- | --- |
| Patch density | `data/inputs/patch_density/pd_{year}.tif` | Original aligned PD values; higher values are reverse-oriented |
| Landscape cohesion | `data/inputs/cohesion/cohesion_{year}.tif` | Original aligned COHESION values |
| Vegetation component | `data/inputs/vegetation_resilience/vfr_{year}.tif` | Precomputed VFR, not raw NDVI |
| Hydrological component | `data/inputs/hydrological_resilience/hr_{year}.tif` | Precomputed HR, not raw water frequency or runoff |
| Habitat quality | `data/inputs/habitat_quality/hq_{year}.tif` | Precomputed InVEST HQ |
| Assessment mask | `data/inputs/wetland_mask.tif` | The original study mask: wetland = 1; other values are excluded |
| Ten predictors | `data/inputs/predictors/` | The order and patterns specified below and in `config.json` |

The mask is applied to HQ and WRI **after WRI normalization**, matching the
separate wetland-masking step implied by the original source paths. Supply the
same mask and input spatial extent used for the research. If both HQ and WRI
already have NoData outside that assessment mask, `paths.wetland_mask` may be
set to `null`. Do not disable masking for unrestricted landscape rasters.

The indicator extrema are taken from all finite input indicator pixels across
the configured years, before this classification mask is applied. Changing
their input footprint changes normalization and can change the results.

### Predictor order and scaling

| Order | Predictor name | Default pattern in `data/inputs/predictors/` | Multiplier |
| --- | --- | --- | --- |
| 1 | `actual_evapotranspiration` | `aet_{year}.tif` | 0.1 |
| 2 | `distance_to_coast` | `coast_distance_{year}.tif` | 1 |
| 3 | `climatic_water_deficit` | `water_deficit_{year}.tif` | 0.1 |
| 4 | `elevation` | `elevation_{year}.tif` | 1 |
| 5 | `mean_annual_precipitation` | `precipitation_{year}.tif` | 1 |
| 6 | `mean_annual_temperature` | `temperature_{year}.tif` | 1 |
| 7 | `population` | `population_{year}.tif` | 1 |
| 8 | `built_up_land_proportion` | `built_up_{year}.tif` | 1 |
| 9 | `cropland_proportion` | `cropland_{year}.tif` | 1 |
| 10 | `aquaculture_proportion` | `aquaculture_{year}.tif` | 1 |

The 0.1 multipliers for AET and climatic water deficit reproduce the supplied
preparation script. If your files are already expressed in physical units,
change these multipliers to 1 to avoid applying the scale twice. Other unit
conversions, aggregation, and land-use proportions must already be represented
in the input rasters. The source does not establish whether `population` is a
count or density; retain the actual research variable and document its units
with the data release. Static rasters can use paths without `{year}`.

## Run the workflow

From this repository's directory:

```shell
python 01_compute_wetland_resilience.py --config config.json
python 02_classify_habitat_resilience.py --config config.json
python 03_prepare_training_data.py --config config.json
python 04_train_xgboost_random_search.py --config config.json
```

The first two steps use **all configured years** because pooled normalization
and, optionally, thresholds depend on that set. For preparation and training,
a subset can be selected without altering the preceding normalization:

```shell
python 03_prepare_training_data.py --config config.json --years 2002
python 04_train_xgboost_random_search.py --config config.json --years 2002
```

Existing WRI rasters can be supplied by editing `paths.wri` and skipping step
1. Existing mismatch rasters can be supplied through `paths.mismatch` and
skipping steps 1 and 2, provided their class codes match those below. To train
from earlier `.npy` arrays, matching feature-order and split metadata are also
required; the new preparation script writes both automatically.

For mismatch rasters produced by step 2, preparation checks the saved
classification configuration before using them. External mismatch rasters
without that metadata are flagged as unverified inputs; their threshold
provenance must be confirmed separately. Selecting a classification mode in
the configuration does not recalculate existing class rasters.

Each stage refuses to overwrite a nonempty output directory. To rerun an
analysis, point its outputs to a **new directory**. Preserve earlier runs
separately when changing classification, predictors, or sampling settings.

### WRI and mismatch classes

Each component is normalized using its pooled minimum and maximum. PD is
reverse-oriented. The final index is:

```text
WRI = (PD_reversed_normalized * COHESION_normalized
       * VFR_normalized * HR_normalized) ** 0.25
```

All four components must be valid. A zero component yields WRI = 0. Constant
or entirely missing pooled indicator ranges raise an error rather than being
silently assigned a score.

| Raster code | State | Binary model response |
| --- | --- | --- |
| 1 | HH: high HQ, high WRI | 0 |
| 2 | HL: high HQ, low WRI | 1 |
| 3 | LH: low HQ, high WRI | 0 |
| 4 | LL: low HQ, low WRI | 0 |

Values equal to a threshold belong to the high class. Output raster NoData is
`-9999`, written as float32 with LZW compression. Invalid labels are not
silently converted to negative-class samples.

### Randomized XGBoost search

- Blocks are 20 by 20 **pixels**, numbered from the top-left of the input grid;
  the physical size follows the raster's coordinate system and resolution.
- Approximately 80% of occupied blocks are assigned to training by a seeded
  shuffle; the remaining blocks are held out. Pixel counts need not be 80:20.
- The training blocks undergo five-fold cross-validation. All samples in a
  block remain together. No additional stratification, buffer, or resampling
  is introduced.
- Fifty distinct combinations are sampled from 5,184 combinations, using the
  same candidate list in all years. Maximum mean validation ROC AUC selects
  the model; log loss is the internal evaluation metric.
- The negative/positive class ratio from the full annual training set is used
  in every fold and final fit. The selected model is refitted on the full
  training set without early stopping.
- Held-out metrics use probabilities for ROC AUC and a threshold of 0.5 for
  classification. Importance is normalized mean split gain.
- Seed 42 is retained for reproducibility. The block shuffle uses NumPy
  `RandomState`, preserving the original sequence rather than replacing it
  with a different random-number generator.

A full default run makes 251 fits per year, or 1,506 fits for six years.
Use the synthetic tests below to check installation without launching this
full analysis.

## Outputs created when the scripts run

| Directory | Contents |
| --- | --- |
| `outputs/wri/` | Annual WRI GeoTIFFs and pooled-normalization metadata |
| `outputs/mismatch/` | Annual class rasters, thresholds, counts, and configuration |
| `outputs/training_data/<year>/` | Eight NumPy arrays, feature names, and split metadata |
| `outputs/models/<year>/` | Native XGBoost model, candidate and fold results, selected parameters, test metrics, predictions, importance, and run metadata |

The model format is native XGBoost JSON rather than Python pickle:

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("outputs/models/2002/model.json")
```

## Tests

```shell
python -B -m unittest discover -s tests -v
```

Tests generate small synthetic inputs in temporary directories and use only
a few shallow trees. They verify numerical logic and workflow operation, not
the manuscript's full-data results. No research-data run is required and no
test outputs are retained in `outputs/`.

