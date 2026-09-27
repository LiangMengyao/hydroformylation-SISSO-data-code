#!/usr/bin/env python3
"""Hard gate: reproduce both frozen full-data models before any LOOCV run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from model_tools import normalize_expression, parse_models
from workflow import (
    denominator_diagnostics,
    local_feature_rows,
    model_record,
    predict_model,
    read_dicts,
    reconstruct_models,
    run_sisso,
    target_value,
    write_dicts,
)


ROOT = Path(__file__).resolve().parents[1]


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def reference_d4(kind: str) -> dict[str, object]:
    models = parse_models((ROOT / "reference" / kind / "SISSO.out").read_text(encoding="utf-8", errors="replace"))
    matches = [model for model in models if int(model["dimension"]) == 4]
    if len(matches) != 1:
        raise RuntimeError(f"{kind}: frozen reference does not contain exactly one D4 model")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sisso-exe", type=Path, required=True)
    parser.add_argument("--mpi-tasks", type=int, default=8)
    args = parser.parse_args()
    if not args.sisso_exe.is_file():
        raise FileNotFoundError(args.sisso_exe)

    regression_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    equation_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    all_pass = True
    for kind, target, expected_count in (("mono", "raw", 28), ("bi", "log", 31)):
        rows = read_dicts(ROOT / "input" / f"{kind}_full_data.csv")
        features = [field for field in rows[0] if field not in {"sample", "source_name", "L_B"}]
        selected_rows = read_dicts(ROOT / "full_data" / kind / "selected_features.csv")
        selected = [row["source_descriptor"] for row in selected_rows]
        if len(selected) != expected_count:
            raise RuntimeError(f"{kind}: screened count changed before regression run")
        run_dir = ROOT / "full_data" / kind
        run_sisso(run_dir, args.sisso_exe, args.mpi_tasks)
        models = reconstruct_models(run_dir, rows, selected, target)
        model = models[4]
        reference = reference_d4(kind)
        expressions_match = [normalize_expression(str(value)) for value in model["expressions"]] == [
            normalize_expression(str(value)) for value in reference["expressions"]
        ]
        coefficient_difference = max(
            relative_difference(float(a), float(b))
            for a, b in zip(model["sisso_coefficients"], reference["sisso_coefficients"])
        )
        intercept_difference = relative_difference(
            float(model["sisso_intercept"]), float(reference["sisso_intercept"])
        )
        rmse_difference = abs(float(model["reported_rmse"]) - float(reference["reported_rmse"]))
        mapping = {f"feature{index}": feature for index, feature in enumerate(selected, start=1)}
        diagnostics = denominator_diagnostics("full_data", model, rows, None, selected, mapping)
        sign_crossing_count = sum(bool(row["sign_crossing_on_train"]) for row in diagnostics)
        mono_crossing_proof = kind != "mono" or sign_crossing_count >= 1
        passed = (
            expressions_match
            and coefficient_difference <= 2e-7
            and intercept_difference <= 2e-7
            and rmse_difference <= 2e-7
            and mono_crossing_proof
        )
        all_pass = all_pass and passed
        regression_rows.append({
            "ligand_class": kind,
            "target": target,
            "screened_feature_count": len(selected),
            "Phi": 2,
            "D": 4,
            "operators": "()(+)(-)(/)(^-1)(^2)(sqrt)(|-|)",
            "safe_denominator": False,
            "expressions_match_reference": expressions_match,
            "max_relative_SISSO_coefficient_difference": coefficient_difference,
            "relative_SISSO_intercept_difference": intercept_difference,
            "absolute_reported_RMSE_difference": rmse_difference,
            "sign_crossing_denominator_count": sign_crossing_count,
            "nested_LOOCV_mono_crossing_proof": mono_crossing_proof,
            "status": "PASS" if passed else "FAIL",
            "source_binary": str(args.sisso_exe),
        })
        values = local_feature_rows(rows, selected)
        for row, local, prediction in zip(rows, values, model["training_predictions"]):
            direct_prediction = predict_model(model, local)
            if abs(float(prediction) - direct_prediction) > 1e-9 * max(1.0, abs(float(prediction))):
                raise RuntimeError(f"{kind}/{row['sample']}: independent equation reconstruction failed")
            prediction_rows.append({
                "ligand_class": kind,
                "sample": row["sample"],
                "target": target,
                "y_true_model": target_value(row["L_B"], target),
                "y_pred_model": direct_prediction,
                "residual_model": direct_prediction - target_value(row["L_B"], target),
                "y_true_LB": float(row["L_B"]),
                "y_pred_LB": direct_prediction if target == "raw" else math.exp(direct_prediction),
                "residual_LB": (direct_prediction if target == "raw" else math.exp(direct_prediction)) - float(row["L_B"]),
            })
        equation_rows.append({
            "analysis": "full_data",
            "ligand_class": kind,
            "target": target,
            "screened_feature_count": len(selected),
            "Phi": 2,
            **model_record(model, mapping),
        })
        diagnostic_rows.extend({"ligand_class": kind, "target": target, **row} for row in diagnostics)

    result_dir = ROOT / "results"
    write_dicts(result_dir / "full_data_model_regression_check.csv", regression_rows, list(regression_rows[0]))
    write_dicts(result_dir / "full_data_training_predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_dicts(result_dir / "full_data_equations.csv", equation_rows, list(equation_rows[0]))
    diagnostic_fields = list(diagnostic_rows[0]) if diagnostic_rows else ["ligand_class", "target", "analysis"]
    write_dicts(result_dir / "full_data_denominator_diagnostics.csv", diagnostic_rows, diagnostic_fields)
    marker = result_dir / "full_data_regression.ok"
    if marker.exists():
        marker.unlink()
    if not all_pass:
        raise RuntimeError("FULL_DATA_REGRESSION: FAIL; LOOCV must not be started")
    marker.write_text("FULL_DATA_REGRESSION_PASS\n", encoding="ascii")
    print("FULL_DATA_REGRESSION: PASS")
    print("- monodentate raw L/B D4 equation matches frozen manuscript model")
    print("- bidentate ln(L/B) D4 equation matches frozen B2 nested_LOOCV model")
    print("- original sign-crossing monodentate denominator is present")


if __name__ == "__main__":
    main()

