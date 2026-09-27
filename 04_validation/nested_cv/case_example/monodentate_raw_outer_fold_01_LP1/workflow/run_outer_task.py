#!/usr/bin/env python3
"""Run fixed-specification and fully nested nested_LOOCV LOOCV for one outer fold."""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
from pathlib import Path

from model_tools import metrics
from workflow import (
    DIMENSION_GRID,
    FIXED_DIMENSION,
    FIXED_PHI,
    PHI_GRID,
    SCREENING_FIELDS,
    denominator_diagnostics,
    lb_from_model,
    local_feature_rows,
    mapping_rows,
    model_record,
    ordered_features,
    predict_model,
    read_dicts,
    reconstruct_models,
    run_sisso,
    screen,
    target_value,
    write_dicts,
    write_sisso_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_record(task_id: int) -> dict[str, str]:
    matches = [row for row in read_dicts(ROOT / "cases_manifest.csv") if int(row["task_id"]) == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for task {task_id}, found {len(matches)}")
    return matches[0]


def source_features(rows: list[dict[str, str]]) -> list[str]:
    return [field for field in rows[0] if field not in {"sample", "source_name", "L_B"}]


def mapping_dict(selected: list[str]) -> dict[str, str]:
    return {f"feature{index}": feature for index, feature in enumerate(selected, start=1)}


def prune_ephemeral(run_dir: Path) -> None:
    for name in ("feature_space", "SIS_subspaces"):
        path = run_dir / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def run_screened_phi_grid(
    run_parent: Path,
    training: list[dict[str, str]],
    validation: list[dict[str, str]],
    all_features: list[str],
    ligand_class: str,
    target: str,
    executable: Path,
    mpi_tasks: int,
    audit_prefix: str,
) -> tuple[dict[tuple[int, int], dict[str, object]], list[dict[str, object]]]:
    if len(training) + len(validation) != 23 and audit_prefix.startswith("inner"):
        raise RuntimeError("Inner split must partition the 23 outer-training ligands")
    selected_set, screening = screen(training, all_features, target)
    selected = ordered_features(selected_set)
    if not selected:
        raise RuntimeError("Screening selected no features")
    write_dicts(run_parent / f"{audit_prefix}_screening.csv", screening, SCREENING_FIELDS)
    write_dicts(
        run_parent / f"{audit_prefix}_selected_features.csv",
        mapping_rows(selected, all_features),
        ["model_feature", "source_descriptor", "source_index", "ordering_family", "mono_unit_group"],
    )
    candidates: dict[tuple[int, int], dict[str, object]] = {}
    qa_rows: list[dict[str, object]] = []
    validation_values = local_feature_rows(validation, selected)
    for phi in PHI_GRID:
        run_dir = run_parent / f"phi{phi}"
        write_sisso_inputs(run_dir, training, selected, all_features, ligand_class, target, phi)
        run_sisso(run_dir, executable, mpi_tasks)
        models = reconstruct_models(run_dir, training, selected, target)
        prune_ephemeral(run_dir)
        for dimension, model in models.items():
            predictions = [predict_model(model, values) for values in validation_values]
            candidates[(phi, dimension)] = {
                "model": model,
                "selected": selected,
                "mapping": mapping_dict(selected),
                "screened_feature_count": len(selected),
                "predictions": predictions,
            }
            qa_rows.append({
                "audit_prefix": audit_prefix,
                "fcomplexity": phi,
                "dimension": dimension,
                "n_training": len(training),
                "screened_feature_count": len(selected),
                "max_descriptor_relative_difference": model["max_descriptor_relative_difference"],
                "refit_vs_reported_RMSE_difference": model["refit_vs_reported_rmse_difference"],
                "normal_equation_pivot_ratio": model["normal_equation_pivot_ratio"],
                "status": "PASS",
            })
    return candidates, qa_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--sisso-exe", type=Path, required=True)
    parser.add_argument("--mpi-tasks", type=int, default=8)
    args = parser.parse_args()

    if not (ROOT / "results" / "full_data_regression.ok").exists():
        raise RuntimeError("Full-data regression gate has not passed; refusing to run LOOCV")
    record = manifest_record(args.task_id)
    case_dir = ROOT / record["case_dir"]
    marker = case_dir / "outer_complete.ok"
    if marker.exists():
        print(f"SKIP_COMPLETE: task={args.task_id} case={record['case_dir']}")
        return
    if not args.sisso_exe.is_file():
        raise FileNotFoundError(args.sisso_exe)

    target = record["target"]
    ligand_class = record["ligand_class"]
    pipeline = record["pipeline"]
    heldout_id = record["heldout_ligand"]
    outer_training = read_dicts(case_dir / "outer_training.csv")
    heldout_rows = read_dicts(case_dir / "heldout_features.csv")
    if len(outer_training) != 23 or len(heldout_rows) != 1:
        raise RuntimeError("Invalid outer split sizes")
    heldout = heldout_rows[0]
    if "L_B" in heldout or heldout["sample"] != heldout_id:
        raise RuntimeError("Held-out target leakage or held-out ID mismatch")
    if heldout_id in {row["sample"] for row in outer_training}:
        raise RuntimeError("Held-out ligand appears in outer training")
    features = source_features(outer_training)

    split_rows = read_dicts(case_dir / "inner_splits.csv")
    assignments = {row["ligand"]: int(row["inner_validation_fold"]) for row in split_rows}
    if set(assignments) != {row["sample"] for row in outer_training}:
        raise RuntimeError("Inner assignments do not match outer-training IDs")

    pooled: dict[tuple[int, int], list[dict[str, object]]] = {
        (phi, dimension): [] for phi in PHI_GRID for dimension in DIMENSION_GRID
    }
    qa_rows: list[dict[str, object]] = []
    for fold in (1, 2, 3):
        inner_validation = [row for row in outer_training if assignments[row["sample"]] == fold]
        inner_training = [row for row in outer_training if assignments[row["sample"]] != fold]
        train_ids = {row["sample"] for row in inner_training}
        validation_ids = {row["sample"] for row in inner_validation}
        if not inner_validation or train_ids & validation_ids or heldout_id in train_ids | validation_ids:
            raise RuntimeError(f"Invalid or leaking inner fold {fold}")
        fold_parent = case_dir / "runs" / f"inner_fold{fold}"
        fold_parent.mkdir(parents=True, exist_ok=True)
        (fold_parent / "inner_training_ligand_ids.txt").write_text(
            "\n".join(row["sample"] for row in inner_training) + "\n", encoding="utf-8"
        )
        (fold_parent / "inner_validation_ligand_ids.txt").write_text(
            "\n".join(row["sample"] for row in inner_validation) + "\n", encoding="utf-8"
        )
        candidates, checks = run_screened_phi_grid(
            fold_parent,
            inner_training,
            inner_validation,
            features,
            ligand_class,
            target,
            args.sisso_exe,
            args.mpi_tasks,
            f"inner_fold{fold}",
        )
        qa_rows.extend(checks)
        for candidate, payload in candidates.items():
            phi, dimension = candidate
            for row, prediction in zip(inner_validation, payload["predictions"]):
                true_model = target_value(row["L_B"], target)
                predicted_lb = lb_from_model(prediction, target)
                pooled[candidate].append({
                    "task_id": args.task_id,
                    "pipeline": pipeline,
                    "outer_heldout_ligand": heldout_id,
                    "inner_validation_fold": fold,
                    "inner_validation_ligand": row["sample"],
                    "fcomplexity": phi,
                    "dimension": dimension,
                    "screened_feature_count": payload["screened_feature_count"],
                    "y_true_model": true_model,
                    "y_pred_model": prediction,
                    "residual_model": prediction - true_model,
                    "y_true_LB": float(row["L_B"]),
                    "y_pred_LB": predicted_lb,
                    "residual_LB": predicted_lb - float(row["L_B"]),
                })

    inner_rows: list[dict[str, object]] = []
    candidate_metric_rows: list[dict[str, object]] = []
    for (phi, dimension), rows in sorted(pooled.items()):
        if len(rows) != 23 or len({row["inner_validation_ligand"] for row in rows}) != 23:
            raise RuntimeError(f"Incomplete pooled inner predictions for Phi={phi}, D={dimension}")
        inner_rows.extend(rows)
        model_metrics = metrics(
            [float(row["y_true_model"]) for row in rows],
            [float(row["y_pred_model"]) for row in rows],
        )
        lb_metrics = metrics(
            [float(row["y_true_LB"]) for row in rows],
            [float(row["y_pred_LB"]) for row in rows],
        )
        candidate_metric_rows.append({
            "task_id": args.task_id,
            "pipeline": pipeline,
            "outer_heldout_ligand": heldout_id,
            "target": target,
            "fcomplexity": phi,
            "dimension": dimension,
            "n_inner_oof": 23,
            "pooled_model_R2": model_metrics["R2"],
            "pooled_model_RMSE": model_metrics["RMSE"],
            "pooled_model_MAE": model_metrics["MAE"],
            "pooled_model_MaxAE": model_metrics["MaxAE"],
            "pooled_LB_R2": lb_metrics["R2"],
            "pooled_LB_RMSE": lb_metrics["RMSE"],
            "pooled_LB_MAE": lb_metrics["MAE"],
            "pooled_LB_MaxAE": lb_metrics["MaxAE"],
        })
    selected_metric = min(
        candidate_metric_rows,
        key=lambda row: (
            float(row["pooled_model_RMSE"]),
            float(row["pooled_model_MAE"]),
            float(row["pooled_model_MaxAE"]),
            int(row["dimension"]),
            int(row["fcomplexity"]),
        ),
    )
    selected_phi = int(selected_metric["fcomplexity"])
    selected_dimension = int(selected_metric["dimension"])

    outer_parent = case_dir / "runs" / "outer_refit"
    selected_set, outer_screening = screen(outer_training, features, target)
    outer_selected = ordered_features(selected_set)
    write_dicts(case_dir / "outer_screening.csv", outer_screening, SCREENING_FIELDS)
    write_dicts(
        case_dir / "outer_selected_features.csv",
        mapping_rows(outer_selected, features),
        ["model_feature", "source_descriptor", "source_index", "ordering_family", "mono_unit_group"],
    )
    outer_mapping = mapping_dict(outer_selected)
    outer_models: dict[int, dict[int, dict[str, object]]] = {}
    for phi in sorted({FIXED_PHI, selected_phi}):
        run_dir = outer_parent / f"phi{phi}"
        write_sisso_inputs(run_dir, outer_training, outer_selected, features, ligand_class, target, phi)
        run_sisso(run_dir, args.sisso_exe, args.mpi_tasks)
        outer_models[phi] = reconstruct_models(run_dir, outer_training, outer_selected, target)
        prune_ephemeral(run_dir)
        for dimension, model in outer_models[phi].items():
            qa_rows.append({
                "audit_prefix": "outer_refit",
                "fcomplexity": phi,
                "dimension": dimension,
                "n_training": 23,
                "screened_feature_count": len(outer_selected),
                "max_descriptor_relative_difference": model["max_descriptor_relative_difference"],
                "refit_vs_reported_RMSE_difference": model["refit_vs_reported_rmse_difference"],
                "normal_equation_pivot_ratio": model["normal_equation_pivot_ratio"],
                "status": "PASS",
            })

    fixed_model = outer_models[FIXED_PHI][FIXED_DIMENSION]
    nested_model = outer_models[selected_phi][selected_dimension]
    heldout_values = local_feature_rows([heldout], outer_selected)[0]
    fixed_prediction = predict_model(fixed_model, heldout_values)
    nested_prediction = predict_model(nested_model, heldout_values)
    baseline_prediction = statistics.fmean(target_value(row["L_B"], target) for row in outer_training)

    fixed_row = {
        "task_id": args.task_id,
        "pipeline": pipeline,
        "ligand_class": ligand_class,
        "target": target,
        "outer_fold": record["outer_fold"],
        "heldout_ligand": heldout_id,
        "screened_feature_count": len(outer_selected),
        "Phi": FIXED_PHI,
        "D": FIXED_DIMENSION,
        "y_pred_model": fixed_prediction,
        "y_pred_LB": lb_from_model(fixed_prediction, target),
        "negative_raw_prediction": target == "raw" and fixed_prediction < 0,
    }
    nested_row = {
        "task_id": args.task_id,
        "pipeline": pipeline,
        "ligand_class": ligand_class,
        "target": target,
        "outer_fold": record["outer_fold"],
        "heldout_ligand": heldout_id,
        "screened_feature_count": len(outer_selected),
        "selected_Phi": selected_phi,
        "selected_D": selected_dimension,
        "selected_inner_RMSE": selected_metric["pooled_model_RMSE"],
        "y_pred_model": nested_prediction,
        "y_pred_LB": lb_from_model(nested_prediction, target),
        "negative_raw_prediction": target == "raw" and nested_prediction < 0,
    }
    baseline_row = {
        "task_id": args.task_id,
        "pipeline": pipeline,
        "ligand_class": ligand_class,
        "target": target,
        "outer_fold": record["outer_fold"],
        "heldout_ligand": heldout_id,
        "y_pred_model": baseline_prediction,
        "y_pred_LB": lb_from_model(baseline_prediction, target),
    }
    write_dicts(case_dir / "fixed_spec_prediction.csv", [fixed_row], list(fixed_row))
    write_dicts(case_dir / "nested_prediction.csv", [nested_row], list(nested_row))
    write_dicts(case_dir / "mean_baseline_prediction.csv", [baseline_row], list(baseline_row))

    equation_rows = [
        {
            "task_id": args.task_id,
            "analysis": "fixed_spec",
            "pipeline": pipeline,
            "ligand_class": ligand_class,
            "target": target,
            "outer_fold": record["outer_fold"],
            "heldout_ligand": heldout_id,
            "screened_feature_count": len(outer_selected),
            "Phi": FIXED_PHI,
            **model_record(fixed_model, outer_mapping),
        },
        {
            "task_id": args.task_id,
            "analysis": "nested",
            "pipeline": pipeline,
            "ligand_class": ligand_class,
            "target": target,
            "outer_fold": record["outer_fold"],
            "heldout_ligand": heldout_id,
            "screened_feature_count": len(outer_selected),
            "Phi": selected_phi,
            **model_record(nested_model, outer_mapping),
        },
    ]
    write_dicts(case_dir / "fold_equations.csv", equation_rows, list(equation_rows[0]))

    diagnostics: list[dict[str, object]] = []
    for analysis, phi, dimension, model in (
        ("fixed_spec", FIXED_PHI, FIXED_DIMENSION, fixed_model),
        ("nested", selected_phi, selected_dimension, nested_model),
    ):
        for diagnostic in denominator_diagnostics(
            analysis, model, outer_training, heldout, outer_selected, outer_mapping
        ):
            diagnostics.append({
                "task_id": args.task_id,
                "pipeline": pipeline,
                "ligand_class": ligand_class,
                "target": target,
                "outer_fold": record["outer_fold"],
                "heldout_ligand": heldout_id,
                "Phi": phi,
                "D": dimension,
                **diagnostic,
            })
    diagnostic_fields = list(diagnostics[0]) if diagnostics else [
        "task_id", "pipeline", "ligand_class", "target", "outer_fold", "heldout_ligand", "Phi", "D", "analysis"
    ]
    write_dicts(case_dir / "denominator_diagnostics.csv", diagnostics, diagnostic_fields)
    write_dicts(case_dir / "inner_oof_predictions.csv", inner_rows, list(inner_rows[0]))
    write_dicts(case_dir / "inner_candidate_metrics.csv", candidate_metric_rows, list(candidate_metric_rows[0]))
    write_dicts(case_dir / "model_reconstruction_checks.csv", qa_rows, list(qa_rows[0]))
    status = {
        "task_id": args.task_id,
        "pipeline": pipeline,
        "ligand_class": ligand_class,
        "heldout_ligand": heldout_id,
        "fixed_spec_complete": True,
        "nested_complete": True,
        "inner_candidate_count": len(candidate_metric_rows),
        "selected_Phi": selected_phi,
        "selected_D": selected_dimension,
        "safe_denominator": False,
        "manual_prediction_replacement": False,
        "prediction_clipping": False,
        "status": "PASS",
    }
    write_dicts(case_dir / "run_status.csv", [status], list(status))
    marker.write_text("OUTER_COMPLETE\n", encoding="ascii")
    print(
        f"OUTER_OK: task={args.task_id} pipeline={pipeline} heldout={heldout_id} "
        f"fixed={fixed_prediction:.12g} nested={nested_prediction:.12g} "
        f"selected_Phi={selected_phi} selected_D={selected_dimension}"
    )


if __name__ == "__main__":
    main()

