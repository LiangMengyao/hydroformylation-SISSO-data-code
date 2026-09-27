#!/usr/bin/env python3
"""Collect and validate the final nested_LOOCV fixed and nested LOOCV outputs."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from model_tools import metrics
from workflow import read_dicts, write_dicts


ROOT = Path(__file__).resolve().parents[1]


def append_rows(destination: list[dict[str, object]], path: Path) -> None:
    rows = read_dicts(path)
    if not rows:
        raise RuntimeError(f"Required result is empty: {path}")
    destination.extend(rows)


def add_truth(rows: list[dict[str, object]], truth: dict[tuple[str, str], float]) -> None:
    for row in rows:
        key = (str(row["pipeline"]), str(row["heldout_ligand"]))
        if key not in truth:
            raise RuntimeError(f"Missing held-out truth for {key}")
        true_lb = truth[key]
        target = str(row["target"])
        true_model = true_lb if target == "raw" else math.log(true_lb)
        pred_model = float(row["y_pred_model"])
        pred_lb = float(row["y_pred_LB"])
        if target == "log" and abs(math.exp(pred_model) - pred_lb) > 1e-10 * max(1.0, pred_lb):
            raise RuntimeError(f"Log back-transform mismatch for {key}")
        row["experimental_LB"] = true_lb
        row["y_true_model"] = true_model
        row["residual_model"] = pred_model - true_model
        row["absolute_error_model"] = abs(pred_model - true_model)
        row["residual_LB"] = pred_lb - true_lb
        row["absolute_error_LB"] = abs(pred_lb - true_lb)
        row["relative_error_percent_LB"] = abs(pred_lb - true_lb) / true_lb * 100.0


def validate_prediction_set(rows: list[dict[str, object]], label: str) -> None:
    if len(rows) != 48:
        raise RuntimeError(f"{label}: expected 48 predictions, found {len(rows)}")
    counts = Counter(str(row["ligand_class"]) for row in rows)
    if counts != Counter({"mono": 24, "bi": 24}):
        raise RuntimeError(f"{label}: expected 24 predictions per ligand class, found {counts}")
    keys = [(str(row["pipeline"]), str(row["heldout_ligand"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label}: duplicate held-out predictions")
    if not all(math.isfinite(float(row["y_pred_model"])) and math.isfinite(float(row["y_pred_LB"])) for row in rows):
        raise RuntimeError(f"{label}: non-finite prediction")


def metric_rows(rows: list[dict[str, object]], analysis: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for ligand_class in ("mono", "bi"):
        subset = [row for row in rows if row["ligand_class"] == ligand_class]
        target = str(subset[0]["target"])
        if target == "log":
            result = metrics(
                [float(row["y_true_model"]) for row in subset],
                [float(row["y_pred_model"]) for row in subset],
            )
            output.append({
                "analysis": analysis,
                "ligand_class": ligand_class,
                "target": target,
                "scale": "log",
                "n": len(subset),
                **result,
                "negative_prediction_count": 0,
            })
        result = metrics(
            [float(row["experimental_LB"]) for row in subset],
            [float(row["y_pred_LB"]) for row in subset],
        )
        output.append({
            "analysis": analysis,
            "ligand_class": ligand_class,
            "target": target,
            "scale": "L_B",
            "n": len(subset),
            **result,
            "negative_prediction_count": sum(float(row["y_pred_LB"]) < 0 for row in subset),
        })
    return output


def main() -> None:
    manifest = read_dicts(ROOT / "cases_manifest.csv")
    if len(manifest) != 48:
        raise RuntimeError("Manifest no longer contains exactly 48 outer folds")
    truth_rows = read_dicts(ROOT / "truth" / "heldout_truth.csv")
    truth = {
        (row["pipeline"], row["heldout_ligand"]): float(row["L_B"])
        for row in truth_rows
    }
    fixed: list[dict[str, object]] = []
    nested: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    inner_metrics: list[dict[str, object]] = []
    denominator_rows: list[dict[str, object]] = []
    equation_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    for record in manifest:
        case_dir = ROOT / record["case_dir"]
        if not (case_dir / "outer_complete.ok").exists():
            raise RuntimeError(f"Outer fold is incomplete: {record['case_dir']}")
        append_rows(fixed, case_dir / "fixed_spec_prediction.csv")
        append_rows(nested, case_dir / "nested_prediction.csv")
        append_rows(baseline, case_dir / "mean_baseline_prediction.csv")
        append_rows(inner_metrics, case_dir / "inner_candidate_metrics.csv")
        if (case_dir / "denominator_diagnostics.csv").stat().st_size:
            denominator_rows.extend(read_dicts(case_dir / "denominator_diagnostics.csv"))
        append_rows(equation_rows, case_dir / "fold_equations.csv")
        append_rows(status_rows, case_dir / "run_status.csv")
        append_rows(reconstruction_rows, case_dir / "model_reconstruction_checks.csv")
    for label, rows in (("fixed", fixed), ("nested", nested), ("baseline", baseline)):
        add_truth(rows, truth)
        validate_prediction_set(rows, label)
    if len(inner_metrics) != 48 * 8:
        raise RuntimeError(f"Expected 384 nested inner-candidate rows, found {len(inner_metrics)}")
    if len(equation_rows) != 96 or len(status_rows) != 48:
        raise RuntimeError("Fold-equation or run-status row count mismatch")
    if not all(row["status"] == "PASS" for row in status_rows):
        raise RuntimeError("At least one outer-fold run status is not PASS")

    result_dir = ROOT / "results"
    write_dicts(result_dir / "fixed_spec_loocv_predictions.csv", fixed, list(fixed[0]))
    fixed_metrics = metric_rows(fixed, "fixed_spec")
    write_dicts(result_dir / "fixed_spec_loocv_metrics.csv", fixed_metrics, list(fixed_metrics[0]))
    write_dicts(result_dir / "nested_loocv_predictions.csv", nested, list(nested[0]))
    nested_metrics = metric_rows(nested, "nested")
    write_dicts(result_dir / "nested_loocv_metrics.csv", nested_metrics, list(nested_metrics[0]))
    write_dicts(result_dir / "mean_baseline_predictions.csv", baseline, list(baseline[0]))
    baseline_metrics = metric_rows(baseline, "mean_baseline")
    write_dicts(result_dir / "mean_baseline_metrics.csv", baseline_metrics, list(baseline_metrics[0]))
    write_dicts(result_dir / "nested_inner_selection_summary.csv", inner_metrics, list(inner_metrics[0]))

    frequency = Counter(
        (row["ligand_class"], int(row["selected_Phi"]), int(row["selected_D"]))
        for row in nested
    )
    frequency_rows = [
        {"ligand_class": kind, "Phi": phi, "D": dimension, "count": count, "frequency": count / 24.0}
        for (kind, phi, dimension), count in sorted(frequency.items())
    ]
    write_dicts(result_dir / "nested_complexity_frequency.csv", frequency_rows, list(frequency_rows[0]))
    diagnostic_fields = list(denominator_rows[0]) if denominator_rows else ["analysis", "ligand_class"]
    write_dicts(result_dir / "denominator_diagnostics.csv", denominator_rows, diagnostic_fields)
    write_dicts(result_dir / "fold_equations.csv", equation_rows, list(equation_rows[0]))
    write_dicts(result_dir / "run_status.csv", status_rows, list(status_rows[0]))
    write_dicts(result_dir / "model_reconstruction_checks.csv", reconstruction_rows, list(reconstruction_rows[0]))
    (result_dir / "collection.ok").write_text("COLLECT_OK\n", encoding="ascii")
    print("COLLECT_OK")
    print("- fixed-specification predictions: 48")
    print("- fully nested predictions: 48")
    print("- fold-matched mean baseline predictions: 48")
    print("- nested inner candidate records: 384")


if __name__ == "__main__":
    main()

