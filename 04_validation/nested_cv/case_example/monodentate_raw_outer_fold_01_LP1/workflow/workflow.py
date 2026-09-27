#!/usr/bin/env python3
"""Shared utilities for the frozen nested_LOOCV SISSO validation workflow."""

from __future__ import annotations

import csv
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from model_tools import (
    ExpressionParser,
    mapped_expression,
    metrics,
    normalize_expression,
    parse_models,
    read_desc_file,
    refit_same_expression,
)


TOP_N = 20
NF_SIS = 100
OPS = "()(+)(-)(/)(^-1)(^2)(sqrt)(|-|)"
PHI_GRID = (1, 2)
DIMENSION_GRID = (1, 2, 3, 4)
FIXED_PHI = 2
FIXED_DIMENSION = 4


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_dicts(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {label}: {value}")
    return value


def target_value(lb: str | float, target: str) -> float:
    value = finite(float(lb), "L/B")
    if value <= 0:
        raise ValueError(f"L/B must be positive, found {value}")
    return value if target == "raw" else math.log(value)


def lb_from_model(value: float, target: str) -> float:
    if target == "raw":
        return finite(value, "raw L/B prediction")
    try:
        return finite(math.exp(value), "back-transformed L/B prediction")
    except OverflowError as exc:
        raise ValueError(f"Overflow while exponentiating log prediction {value}") from exc


def pearson_r(x: list[float], y: list[float]) -> float:
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    sx = sum((value - mx) ** 2 for value in x)
    sy = sum((value - my) ** 2 for value in y)
    if sx <= 0 or sy <= 0:
        return 0.0
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return numerator / math.sqrt(sx * sy)


def chatterjee_xi(x: list[float], y: list[float]) -> float:
    """Deterministic implementation used by the successful reproduction jobs."""
    order = sorted(range(len(x)), key=lambda index: (x[index], index))
    y_sorted = [y[index] for index in order]
    rank_order = sorted(range(len(y_sorted)), key=lambda index: (y_sorted[index], index))
    ranks = [0] * len(y_sorted)
    for rank, index in enumerate(rank_order):
        ranks[index] = rank
    n = len(y_sorted)
    if n < 2:
        raise ValueError("Chatterjee xi requires at least two samples")
    return 1.0 - 3.0 * sum(abs(ranks[i + 1] - ranks[i]) for i in range(n - 1)) / (n * n - 1.0)


SCREENING_FIELDS = [
    "feature", "source_index", "pearson_r", "pearson_abs", "chatterjee_xi",
    "pearson_top20_rank", "chatterjee_top20_rank", "selected_union",
]


def screen(training: list[dict[str, str]], features: list[str], target: str):
    y = [target_value(row["L_B"], target) for row in training]
    records: list[dict[str, object]] = []
    for source_index, feature in enumerate(features, start=1):
        x = [finite(float(row[feature]), f"{row['sample']}:{feature}") for row in training]
        r = pearson_r(x, y)
        xi = chatterjee_xi(x, y)
        records.append({
            "feature": feature,
            "source_index": source_index,
            "pearson_r": r,
            "pearson_abs": abs(r),
            "chatterjee_xi": xi,
        })
    # Feature-name tie-breaking exactly matches the audited reproduction script.
    pearson_top = sorted(records, key=lambda row: (-float(row["pearson_abs"]), str(row["feature"])))[:TOP_N]
    chatterjee_top = sorted(records, key=lambda row: (-float(row["chatterjee_xi"]), str(row["feature"])))[:TOP_N]
    pearson_rank = {str(row["feature"]): rank for rank, row in enumerate(pearson_top, start=1)}
    chatterjee_rank = {str(row["feature"]): rank for rank, row in enumerate(chatterjee_top, start=1)}
    selected = set(pearson_rank) | set(chatterjee_rank)
    for record in records:
        feature = str(record["feature"])
        record["pearson_top20_rank"] = pearson_rank.get(feature, "")
        record["chatterjee_top20_rank"] = chatterjee_rank.get(feature, "")
        record["selected_union"] = feature in selected
    return selected, records


def order_family(feature: str) -> int:
    if feature.startswith("D"):
        return 1
    if feature in {"ELUMO", "EHOMO", "E(HO-LU)"}:
        return 2
    if feature.startswith("ESP("):
        return 3
    if feature in {"HF\u503c", "Gcorr", "G"}:
        return 4
    if feature.startswith("NPA("):
        return 5
    if feature.startswith("Q("):
        return 6
    if feature in {"Vbur", "C\u6570"}:
        return 7
    if feature in {"vH", "vCO", "\u03bdH", "\u03bdCO"}:
        return 8
    if feature.startswith("\u03b8") or feature.startswith("\u9525\u89d2"):
        return 9
    if feature.startswith("\u5cf0\u9ad8"):
        return 10
    return 11


def ordered_features(selected: set[str]) -> list[str]:
    return sorted(selected, key=lambda feature: (order_family(feature), feature.casefold()))


def mono_unit_group(feature: str) -> str | None:
    family = order_family(feature)
    return {
        1: "distance",
        2: "orbital_energy",
        3: "esp",
        4: "thermochemical_energy",
        5: "npa_charge",
        6: "charge_or_volume",
        7: "charge_or_volume",
        8: "frequency",
        9: "angle",
        10: "peak_height",
    }.get(family)


def format_funit(ligand_class: str, selected: list[str]) -> str:
    if ligand_class == "bi":
        # Preserve the frozen B2 input convention exactly.
        return f"(1:{len(selected)})"
    groups: list[str] = []
    for feature in selected:
        group = mono_unit_group(feature)
        if group and group not in groups:
            groups.append(group)
    ranges: list[str] = []
    for group in groups:
        indices = [i for i, feature in enumerate(selected, start=1) if mono_unit_group(feature) == group]
        if len(indices) >= 2:
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise RuntimeError(f"Non-contiguous monodentate unit group: {group}")
            ranges.append(f"({indices[0]}:{indices[-1]})")
    return "".join(ranges)


def mapping_rows(selected: list[str], all_features: list[str]) -> list[dict[str, object]]:
    return [
        {
            "model_feature": f"feature{index}",
            "source_descriptor": feature,
            "source_index": all_features.index(feature) + 1,
            "ordering_family": order_family(feature),
            "mono_unit_group": mono_unit_group(feature) or "",
        }
        for index, feature in enumerate(selected, start=1)
    ]


def sisso_input(
    nsample: int, nsf: int, funit: str, fmax_max: str, complexity: int
) -> str:
    return f"""ntask=1
scmt=.false.
restart=0
ptype=1
fstore=1
nsample={nsample}
nsf={nsf}
funit={funit}
fmax_min=1e-3
fmax_max={fmax_max}
safe_division=.false.
safe_div_rho=0.000000
ops='{OPS}'
desc_dim=4
fcomplexity={complexity}
fit_intercept=.true.
nf_sis={NF_SIS}
method_so='L0'
metric='RMSE'
nmodel=100
isconvex=(1,1)
bwidth=0.001
"""


def write_sisso_inputs(
    run_dir: Path,
    training: list[dict[str, str]],
    selected: list[str],
    all_features: list[str],
    ligand_class: str,
    target: str,
    complexity: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_features = [f"feature{i}" for i in range(1, len(selected) + 1)]
    with (run_dir / "train.dat").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["materials", "property", *model_features])
        for row in training:
            writer.writerow([
                row["sample"],
                format(target_value(row["L_B"], target), ".17g"),
                *[format(float(row[feature]), ".17g") for feature in selected],
            ])
    write_dicts(
        run_dir / "feature_mapping.csv",
        mapping_rows(selected, all_features),
        ["model_feature", "source_descriptor", "source_index", "ordering_family", "mono_unit_group"],
    )
    (run_dir / "training_ligand_ids.txt").write_text(
        "\n".join(row["sample"] for row in training) + "\n", encoding="utf-8"
    )
    fmax_max = "1e5" if ligand_class == "mono" else "1e6"
    text = sisso_input(
        len(training), len(selected), format_funit(ligand_class, selected), fmax_max, complexity
    )
    if "safe_division=.false." not in text or "exp" in text:
        raise RuntimeError("LOOCV SISSO input invariant failed")
    (run_dir / "SISSO.in").write_text(text, encoding="ascii", newline="\n")


def clean_generated(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    if "LOOCV" not in str(resolved):
        raise RuntimeError(f"Refusing to clean unexpected directory: {resolved}")
    for name in ("Models", "SIS_subspaces", "feature_space"):
        path = run_dir / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    for name in ("CONTINUE", "log", "SISSO.out", "run.ok"):
        path = run_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def run_sisso(run_dir: Path, executable: Path, mpi_tasks: int) -> None:
    marker = run_dir / "run.ok"
    required = [run_dir / "Models" / "data_top1" / f"desc_D{dimension:03d}.dat" for dimension in DIMENSION_GRID]
    if marker.exists() and (run_dir / "SISSO.out").exists() and all(path.exists() for path in required):
        return
    clean_generated(run_dir)
    with (run_dir / "log").open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            ["/usr/bin/mpiexec", "-n", str(mpi_tasks), str(executable)],
            cwd=run_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"SISSO failed with exit code {result.returncode}: {run_dir}")
    log_text = (run_dir / "log").read_text(encoding="utf-8", errors="replace")
    if "SISSO done successfully" not in log_text and "Have a nice day" not in log_text:
        raise RuntimeError(f"SISSO success marker missing: {run_dir}")
    if not (run_dir / "SISSO.out").exists() or not all(path.exists() and path.stat().st_size for path in required):
        raise RuntimeError(f"All-D output is incomplete: {run_dir}")
    marker.write_text("RUN_OK\n", encoding="ascii")


def local_feature_rows(rows: list[dict[str, str]], selected: list[str]) -> list[dict[str, float]]:
    return [
        {
            f"feature{index}": finite(float(row[feature]), f"{row['sample']}:{feature}")
            for index, feature in enumerate(selected, start=1)
        }
        for row in rows
    ]


def reconstruct_models(
    run_dir: Path,
    training: list[dict[str, str]],
    selected: list[str],
    target: str,
) -> dict[int, dict[str, object]]:
    parsed = parse_models((run_dir / "SISSO.out").read_text(encoding="utf-8", errors="replace"))
    parsed_by_dimension = {int(model["dimension"]): model for model in parsed}
    feature_rows = local_feature_rows(training, selected)
    targets = [target_value(row["L_B"], target) for row in training]
    reconstructed: dict[int, dict[str, object]] = {}
    for dimension in DIMENSION_GRID:
        if dimension not in parsed_by_dimension:
            raise RuntimeError(f"D={dimension} model missing from {run_dir / 'SISSO.out'}")
        parsed_model = parsed_by_dimension[dimension]
        expressions = [str(value) for value in parsed_model["expressions"]]
        refit = refit_same_expression(expressions, feature_rows, targets)
        desc_rows = read_desc_file(run_dir / "Models" / "data_top1" / f"desc_D{dimension:03d}.dat")
        if len(desc_rows) != len(training):
            raise RuntimeError(f"D={dimension} descriptor output row count mismatch")
        max_relative = 0.0
        nodes = refit["nodes"]
        for row_index, feature_values in enumerate(feature_rows):
            for descriptor_index, node in enumerate(nodes):
                calculated = node.evaluate_one(feature_values)
                printed = desc_rows[row_index][3 + descriptor_index]
                difference = abs(calculated - printed) / max(1.0, abs(calculated), abs(printed))
                max_relative = max(max_relative, difference)
        if max_relative > 2e-8:
            raise RuntimeError(
                f"Expression parser mismatch for D={dimension}: relative descriptor difference={max_relative}"
            )
        calculated_rmse = float(refit["training_metrics"]["RMSE"])
        reported_rmse = float(parsed_model["reported_rmse"])
        rmse_difference = abs(calculated_rmse - reported_rmse)
        if rmse_difference > max(2e-6, 1e-4 * max(1.0, abs(reported_rmse))):
            raise RuntimeError(
                f"Refit RMSE mismatch for D={dimension}: calculated={calculated_rmse}, reported={reported_rmse}"
            )
        reconstructed[dimension] = {
            **parsed_model,
            "coefficients": refit["coefficients"],
            "intercept": refit["intercept"],
            "training_predictions": refit["training_predictions"],
            "training_metrics": refit["training_metrics"],
            "normal_equation_pivot_ratio": refit["normal_equation_pivot_ratio"],
            "max_descriptor_relative_difference": max_relative,
            "refit_vs_reported_rmse_difference": rmse_difference,
            "coefficient_source": "high_precision_OLS_refit_of_same_SISSO_expression",
        }
    return reconstructed


def predict_model(model: dict[str, object], values: dict[str, float]) -> float:
    nodes = [ExpressionParser(str(expression)).parse() for expression in model["expressions"]]
    descriptors = [node.evaluate_one(values) for node in nodes]
    prediction = float(model["intercept"]) + sum(
        float(coefficient) * descriptor
        for coefficient, descriptor in zip(model["coefficients"], descriptors)
    )
    return finite(prediction, "model prediction")


def model_record(model: dict[str, object], mapping: dict[str, str]) -> dict[str, object]:
    return {
        "dimension": int(model["dimension"]),
        "expressions_model_features": " ; ".join(str(value) for value in model["expressions"]),
        "expressions_source_descriptors": " ; ".join(
            mapped_expression(str(value), mapping) for value in model["expressions"]
        ),
        "coefficients_refit": " ; ".join(f"{float(value):.17g}" for value in model["coefficients"]),
        "intercept_refit": f"{float(model['intercept']):.17g}",
        "coefficients_SISSO_printed": " ; ".join(
            f"{float(value):.17g}" for value in model["sisso_coefficients"]
        ),
        "intercept_SISSO_printed": f"{float(model['sisso_intercept']):.17g}",
        "coefficient_source": model["coefficient_source"],
        "training_R2_refit": model["training_metrics"]["R2"],
        "training_RMSE_refit": model["training_metrics"]["RMSE"],
        "training_MAE_refit": model["training_metrics"]["MAE"],
        "training_MaxAE_refit": model["training_metrics"]["MaxAE"],
        "SISSO_reported_RMSE": model["reported_rmse"],
        "SISSO_reported_MaxAE": model["reported_maxae"],
        "refit_vs_reported_RMSE_difference": model["refit_vs_reported_rmse_difference"],
        "normal_equation_pivot_ratio": model["normal_equation_pivot_ratio"],
    }


def denominator_diagnostics(
    analysis: str,
    model: dict[str, object],
    training_rows: list[dict[str, str]],
    heldout_row: dict[str, str] | None,
    selected: list[str],
    mapping: dict[str, str],
) -> list[dict[str, object]]:
    training_values = local_feature_rows(training_rows, selected)
    heldout_values = local_feature_rows([heldout_row], selected)[0] if heldout_row else None
    diagnostics: list[dict[str, object]] = []
    for descriptor_index, expression in enumerate(model["expressions"], start=1):
        node = ExpressionParser(str(expression)).parse()
        for denominator_index, denominator in enumerate(node.denominator_nodes(), start=1):
            values = denominator.evaluate(training_values)
            signed_base = denominator.left if denominator.kind == "abs" else denominator
            signed_values = signed_base.evaluate(training_values)
            absolute = [abs(value) for value in values]
            median_abs = statistics.median(absolute)
            heldout_value = denominator.evaluate_one(heldout_values) if heldout_values else ""
            diagnostics.append({
                "analysis": analysis,
                "descriptor_index": descriptor_index,
                "denominator_index": denominator_index,
                "denominator_model_features": denominator.render(),
                "denominator_source_descriptors": mapped_expression(denominator.render(), mapping),
                "train_min": min(values),
                "train_max": max(values),
                "train_min_abs": min(absolute),
                "train_median_abs": median_abs,
                "rho": min(absolute) / median_abs if median_abs > 0 else 0.0,
                "signed_base_expression": signed_base.render(),
                "signed_base_train_min": min(signed_values),
                "signed_base_train_max": max(signed_values),
                "sign_crossing_on_train": not (min(signed_values) > 0 or max(signed_values) < 0),
                "heldout_denominator_value": heldout_value,
                "heldout_abs_over_train_median": (
                    abs(float(heldout_value)) / median_abs if heldout_values and median_abs > 0 else ""
                ),
                "heldout_outside_train_range": (
                    not (min(values) <= float(heldout_value) <= max(values)) if heldout_values else ""
                ),
            })
    return diagnostics


def expression_list_equal(a: list[str], b: list[str]) -> bool:
    return [normalize_expression(value) for value in a] == [normalize_expression(value) for value in b]
