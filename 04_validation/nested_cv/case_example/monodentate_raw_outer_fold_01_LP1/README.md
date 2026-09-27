# Example nested-LOOCV case: monodentate LP1

This directory provides one completed high-performance-computing example from
the nested_LOOCV fully nested leave-one-out cross-validation workflow reported
in the manuscript and Supporting Information. It is outer fold 1 of the
monodentate raw-L/B workflow, in which LP1 (PPh3) is held out.

## Verified result

- Workflow: monodentate, raw L/B response
- Outer fold: 1
- Held-out ligand: LP1
- Outer-training ligands: 23
- Features retained by outer-fold screening: 28
- Inner-selected SISSO settings: Phi = 1 and D = 1
- Pooled inner-validation RMSE: 0.28498950519518246
- Experimental L/B: 1.6782142535864
- Outer-held-out predicted L/B: 1.706322902634824

The prediction is identical to the LP1 row in
`../../results/monodentate_raw/nested_loocv_predictions.csv`. The selected
outer-fold equation is the row with `analysis = nested` in
`case/fold_equations.csv`.

## Directory contents

- `submission/02_smoke_task1_debug.slurm`: original SLURM submission script
  used to execute task 1 on the computing cluster.
- `workflow/`: original Python workflow modules called by the SLURM script.
- `case/`: fold metadata, outer-training data, screening records, inner-fold
  splits and candidate metrics, selected equation, reconstruction checks, and
  the outer-held-out prediction.
- `selected_outer_refit/`: actual SISSO input, feature mapping, training data,
  program output, and run log for the selected Phi = 1 outer refit.

## Reuse note

The SLURM script is preserved as executed and therefore contains the original
module names and absolute path to the SISSO executable. These environment-
specific paths must be adapted before use on another cluster. SISSO itself is
not redistributed here. The workflow also expects the complete project
manifest and source tables when rerunning the fold from scratch; this example
is supplied as a transparent, completed case rather than as a standalone SISSO
distribution.
