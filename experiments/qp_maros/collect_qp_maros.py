#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Validate and normalize Maros--Meszaros FA-CP summary rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_INSTANCES = 134
TOLERANCE = 1e-6
TIME_LIMIT = 1000.0
SGM_SHIFT = 10.0
RELEASE_FIELDS = (
    "instance",
    "status",
    "optimal",
    "termination_reason",
    "exit_code",
    "elapsed_sec",
    "runtime_sec",
    "total_iters",
    "rel_primal",
    "rel_dual",
    "rel_gap",
    "primal_obj",
    "dual_obj",
    "final_al_sigma",
    "sigma_update_rule",
    "sigma_update_count",
    "input_file",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audited_solved(row: dict[str, str]) -> bool:
    residuals = [finite(row[name]) for name in ("rel_primal", "rel_dual", "rel_gap")]
    return (
        row["optimal"] == "1"
        and row["termination_reason"].upper() == "OPTIMAL"
        and all(value is not None and value >= 0.0 for value in residuals)
        and max(value for value in residuals if value is not None) <= TOLERANCE
    )


def sgm10(rows: list[dict[str, str]]) -> float:
    charged = [
        min(float(row["runtime_sec"]), TIME_LIMIT)
        if audited_solved(row)
        else TIME_LIMIT
        for row in rows
    ]
    return math.exp(
        sum(math.log(value + SGM_SHIFT) for value in charged) / len(charged)
    ) - SGM_SHIFT


def validate(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != EXPECTED_INSTANCES:
        errors.append(f"expected {EXPECTED_INSTANCES} rows, found {len(rows)}")
    identities = [row.get("instance", "") for row in rows]
    if len(set(identities)) != len(identities) or not all(identities):
        errors.append("instance identities are empty or duplicated")
    for row in rows:
        name = row.get("instance", "<unknown>")
        if row.get("exit_code") != "0":
            errors.append(f"{name}: nonzero wrapper exit code")
        runtime = finite(row.get("runtime_sec", ""))
        if runtime is None or runtime < 0.0:
            errors.append(f"{name}: invalid runtime")
        if row.get("sigma_update_rule") != "guarded":
            errors.append(f"{name}: unexpected sigma update rule")
        reported_optimal = row.get("optimal") == "1"
        status_optimal = row.get("termination_reason", "").upper() == "OPTIMAL"
        if reported_optimal != status_optimal:
            errors.append(f"{name}: optimal flag and termination reason disagree")
        if reported_optimal and not audited_solved(row):
            errors.append(f"{name}: reported optimal but residual exceeds {TOLERANCE:g}")
    return errors


def normalize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {name: row[name] for name in RELEASE_FIELDS[:-1]}
        item["input_file"] = Path(row["mps"]).name
        normalized.append(item)
    return normalized


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        help="optional label stored with the generated aggregate",
    )
    args = parser.parse_args()
    source_run = args.source_run or args.summary.stem

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors = validate(rows)
    solved = sum(audited_solved(row) for row in rows)
    aggregate = {
        "method": "RHR-FA-CP",
        "tolerance": TOLERANCE,
        "instances": len(rows),
        "solved": solved,
        "unsolved": len(rows) - solved,
        "sgm10": sgm10(rows) if rows else math.nan,
        "time_limit_sec": TIME_LIMIT,
        "source_run": source_run,
    }
    audit = {
        "source_run": source_run,
        "source_summary_sha256": sha256(args.summary),
        "expected_instances": EXPECTED_INSTANCES,
        "collected_instances": len(rows),
        "tolerance": TOLERANCE,
        "time_limit_sec": TIME_LIMIT,
        "runtime_definition": "solver-reported runtime_sec",
        "sgm_shift_seconds": SGM_SHIFT,
        "termination_reasons": dict(
            sorted(Counter(row["termination_reason"] for row in rows).items())
        ),
        "sigma_update_rules": sorted({row["sigma_update_rule"] for row in rows}),
        "instances_with_sigma_updates": sum(
            int(row["sigma_update_count"]) > 0 for row in rows
        ),
        "instances_with_positive_final_sigma": sum(
            float(row["final_al_sigma"]) > 0.0 for row in rows
        ),
        "aggregate": aggregate,
        "errors": errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Maros--Meszaros audit found {len(errors)} errors")
    write_csv(args.output_dir / "runs.csv", normalize(rows))
    write_csv(args.output_dir / "aggregate.csv", [aggregate])
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
