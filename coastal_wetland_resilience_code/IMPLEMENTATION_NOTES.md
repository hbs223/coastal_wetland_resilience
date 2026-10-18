# Implementation notes

## Confirmed analysis settings

This release is a source-code cleanup, not a reanalysis. The supplied annual
preparation script reads the **long-term pooled-threshold** mismatch rasters,
uses **20 by 20 pixel blocks**, and assigns **80% of occupied blocks** to
training. The supplied random-search script then performs five-fold spatial
block cross-validation on those training blocks.

The author has confirmed these settings for the study and this code release:
`classification.threshold_mode = "pooled"` and
`sampling.train_fraction = 0.8`. One pooled HQ threshold and one pooled WRI
threshold are used for all six assessment years. Approximately 80% of occupied
spatial blocks are assigned to training and the remaining blocks to testing.
The manuscript and supporting information should describe this same procedure.


## Threshold modes

The selected pooled mode and the optional annual mode preserve the corresponding
supplied script's valid-pixel logic:

- **Annual:** each year's HQ and WRI means are computed over pixels where
  both variables are valid within the assessment mask.
- **Pooled:** one HQ mean and one WRI mean are computed across all configured
  years, using each variable's own finite wetland pixels. If their missing-data
  patterns differ, these two means do not use an identical pixel set. Final
  classification still requires both values to be valid.

The original long-term classifier converted NaN to an integer before assigning
NoData. That ordering could retain an unintended integer value. This release
keeps NaN until serialization and writes explicit `-9999` NoData. It rejects
unexpected finite class labels when preparing training data.

