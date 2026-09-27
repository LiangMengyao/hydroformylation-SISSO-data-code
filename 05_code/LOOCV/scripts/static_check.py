#!/usr/bin/env python3
"""Static provenance, leakage, and configuration checks requiring no SISSO run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="ascii"))
    if provenance["operators"] != "()(+)(-)(/)(^-1)(^2)(sqrt)(|-|)":
        fail("Unexpected operator set")
    if provenance["safe_denominator"] is not False:
        fail("Safe denominator must be disabled")
    workbook_paths = {
        "mono": ROOT / "input" / "mono_LP_56_2.xlsx",
        "bi": ROOT / "input" / "bi_LPP_59.xlsx",
    }
    for kind, path in workbook_paths.items():
        if sha256(path) != provenance["source_workbook_sha256"][kind]:
            fail(f"Source workbook hash mismatch: {kind}")
    for relative, expected in provenance["reference_sha256"].items():
        if sha256(ROOT / "reference" / relative) != expected:
            fail(f"Frozen reference hash mismatch: {relative}")

    source_specs = {
        "mono": (56, "LP", "raw", 28),
        "bi": (59, "LPP", "log", 31),
    }
    source_rows: dict[str, list[dict[str, str]]] = {}
    for kind, (n_features, prefix, target, selected_count) in source_specs.items():
        rows = read_csv(ROOT / "input" / f"{kind}_full_data.csv")
        source_rows[kind] = rows
        if len(rows) != 24 or [row["sample"] for row in rows] != [f"{prefix}{i}" for i in range(1, 25)]:
            fail(f"{kind}: source IDs or row count changed")
        features = [field for field in rows[0] if field not in {"sample", "source_name", "L_B"}]
        if len(features) != n_features or len(set(features)) != n_features:
            fail(f"{kind}: descriptor schema changed")
        for row in rows:
            values = [float(row["L_B"]), *[float(row[feature]) for feature in features]]
            if float(row["L_B"]) <= 0 or not all(math.isfinite(value) for value in values):
                fail(f"{kind}/{row['sample']}: invalid numeric input")
        full_dir = ROOT / "full_data" / kind
        selected = read_csv(full_dir / "selected_features.csv")
        reference = read_csv(ROOT / "reference" / kind / "feature_mapping.csv")
        if len(selected) != selected_count:
            fail(f"{kind}: full-data screened count changed")
        if [row["source_descriptor"] for row in selected] != [row["source_descriptor"] for row in reference]:
            fail(f"{kind}: full-data selected feature order differs from frozen reference")
        text = (full_dir / "SISSO.in").read_text(encoding="ascii")
        required = ["safe_division=.false.", "safe_div_rho=0.000000", "fcomplexity=2", "desc_dim=4"]
        if not all(value in text for value in required) or "exp" in text:
            fail(f"{kind}: nested_LOOCV SISSO.in invariant failed")
        if target == "log" and any(float(row["L_B"]) <= 0 for row in rows):
            fail("Log response contains a non-positive L/B")

    manifest = read_csv(ROOT / "cases_manifest.csv")
    truth = read_csv(ROOT / "truth" / "heldout_truth.csv")
    if len(manifest) != 48 or len(truth) != 48:
        fail("Expected exactly 48 outer folds and 48 held-out truth rows")
    counts = Counter(row["ligand_class"] for row in manifest)
    if counts != Counter({"mono": 24, "bi": 24}):
        fail(f"Unexpected outer-fold counts: {counts}")
    truth_key = {(row["task_id"], row["heldout_ligand"]) for row in truth}
    for record in manifest:
        case_dir = ROOT / record["case_dir"]
        training = read_csv(case_dir / "outer_training.csv")
        heldout = read_csv(case_dir / "heldout_features.csv")
        splits = read_csv(case_dir / "inner_splits.csv")
        if len(training) != 23 or len(heldout) != 1 or len(splits) != 23:
            fail(f"Task {record['task_id']}: invalid split sizes")
        heldout_id = record["heldout_ligand"]
        train_ids = [row["sample"] for row in training]
        if heldout_id in train_ids or heldout[0]["sample"] != heldout_id:
            fail(f"Task {record['task_id']}: outer leakage")
        if "L_B" in heldout[0]:
            fail(f"Task {record['task_id']}: held-out target leaked into case input")
        if set(train_ids) != {row["ligand"] for row in splits}:
            fail(f"Task {record['task_id']}: inner split IDs differ from outer training")
        fold_counts = Counter(int(row["inner_validation_fold"]) for row in splits)
        if set(fold_counts) != {1, 2, 3} or min(fold_counts.values()) < 7:
            fail(f"Task {record['task_id']}: invalid deterministic inner folds")
        if (record["task_id"], heldout_id) not in truth_key:
            fail(f"Task {record['task_id']}: held-out truth lookup missing")
        if record["operators"] != provenance["operators"] or record["safe_denominator"] != "False":
            fail(f"Task {record['task_id']}: model-family configuration changed")

    print("STATIC_VALIDATION: PASS")
    print("- exact source workbook and frozen-reference SHA256 checks: PASS")
    print("- operator set identical for mono and bi, with no exp: PASS")
    print("- safe denominator disabled in every full-data input: PASS")
    print("- 48 outer folds, 24 per ligand class: PASS")
    print("- target-free held-out inputs and deterministic inner splits: PASS")
    print("- full-data screening reproduces mono 28 and bi-log 31 pools: PASS")


if __name__ == "__main__":
    main()

