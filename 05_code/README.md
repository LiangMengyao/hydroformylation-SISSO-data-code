# Reproducibility code

The `LOOCV` directory is extracted from the final nested-LOOCV archive used to
generate the validation records in
`04_validation/nested_cv/`.

The Python modules implement:

- Pearson/Chatterjee screening inside each outer training set;
- inner-fold response-scale and SISSO hyperparameter comparison;
- outer-fold refitting and held-out prediction;
- same-expression coefficient refitting and reconstruction checks;
- denominator diagnostics and result collection.

The SLURM files provide the full-data gate, one-fold smoke test, chained
outer-fold execution, result collection, and result packaging. The LP1 case
README identifies the exact command and output represented in the repository.

The workflow expects the project-root layout used during the calculation. The
included `input/` files and `cases_manifest.csv` are the audited source inputs;
the saved case records and final result tables are provided separately under
`04_validation/`. The SISSO executable and cluster modules are environment-
specific and are not included.
