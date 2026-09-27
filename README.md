# Intermediate-derived SISSO data and code for Rh-catalyzed hydroformylation

This repository contains the audited data, optimized structures, final SISSO
models, validation records, and the final nested-LOOCV workflow
supporting the manuscript.

## Contents

- `01_data/`: experimental training responses and full-precision primary
  descriptor matrices. The monodentate matrix contains 56 primary descriptors;
  the bidentate matrix contains 59.
- `02_geometries/`: optimized XYZ structures for the 48 training systems and
  five external ligands. Training systems include both 3ea* and 3ee* geometries;
  external systems include the 3ea* structures used for prediction.
- `03_models/`: final full-data correlation-screened models, full-feature
  control models, SISSO inputs and outputs, descriptor mappings, and fitted
  training predictions.
- `04_validation/`: external predictions, descriptor-space records, mean
  baselines, four response-scale nested-LOOCV workflows, denominator-constrained
  controls, and one complete LP1 outer-fold example.
- `05_code/`: the final nested-LOOCV Python workflow and the
  corresponding cluster submission scripts.

## Response and descriptor conventions

The monodentate model is fitted to raw L/B. The bidentate primary model is
fitted to ln(L/B), and predictions are exponentiated before metrics are
reported on the original L/B scale. The four nested workflows also include the
paired raw- and logarithmic-response comparisons reported in SI Table S7.
Descriptor names, units, atom labels, and numerical conventions are defined in
`05_code/DATA_DICTIONARY.csv` and in the Supporting Information.

The symbols Pa and Pe denote the axial and equatorial phosphorus donor sites,
respectively. The repository uses the ligand identifiers in Tables S1 and S2.

## Reproduction scope

The final workflow in `05_code/LOOCV/` performs descriptor
screening and SISSO model selection inside the outer training folds. It uses
the fixed random seed, descriptor pools, operator set, and Phi/dimension grids
reported in the manuscript and SI. The LP1 case under
`04_validation/nested_cv/case_example/` is a completed, independently traceable
example with its submission script, fold inputs, SISSO input/output, and held-
out prediction.

The SISSO executable is not redistributed. The SLURM files retain the original
cluster module names and executable path and must be adapted to the user's
cluster. The two source Excel files used by the workflow are included under
`05_code/LOOCV/input/`; their SHA-256 values are checked by
the workflow.

## Relation to the manuscript

The numerical records in this repository correspond to the revised manuscript
and Supporting Information, including the final equations, the fully nested
LOOCV metrics, the five external ligands, and the four-dimensional descriptor-
space assessment. The repository is a data and code supplement; the detailed
quantum-chemical and experimental procedures remain in the Supporting
Information.

## License and citation

The analysis code is released under the MIT License in `LICENSE-CODE.txt`.
Unless otherwise noted, the original data, optimized geometries, model records,
and computational results are released under CC BY 4.0 as described in
`LICENSE-DATA.txt`. Experimental values reused from previous publications must
also be attributed to their original sources, including Refs. [3] and [13].

The release should be versioned and cited by its GitHub/Zenodo DOI in the
manuscript Data Availability statement.
