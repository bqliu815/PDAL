#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Validate and aggregate random-LP solver summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


METHODS = ("cupdlpx", "rhr_cpal")
FAMILIES = ("easy", "scaled", "neardep", "combo")
FAMILY_LABELS = {
    "easy": "Baseline",
    "scaled": "Ill-scaled",
    "neardep": "Near-dependent",
    "combo": "Hybrid",
}
METHOD_LABELS = {"cupdlpx": r"\texttt{cuPDLPx}", "rhr_cpal": r"\texttt{RHR-CP-AL}"}
RUNNER_PATH = Path(__file__).with_name("run_random_lp.sbatch")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_sha256(path: Path) -> str:
    token = path.read_text(encoding="utf-8").split(maxsplit=1)[0]
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise RuntimeError(f"{path}: invalid SHA256 record")
    return token


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    identities = [row["instance"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise RuntimeError("suite manifest is empty or contains duplicate instances")
    unknown = sorted({row["family"] for row in rows} - set(FAMILIES))
    if unknown:
        raise RuntimeError(f"unknown random-LP families: {unknown}")
    return rows


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    required = (
        "Termination Reason",
        "Iterations Count",
        "Runtime (sec)",
        "Relative Primal Residual",
        "Relative Dual Residual",
        "Relative Objective Gap",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(f"{path}: missing fields {missing}")
    return values


def parse_cpal_protocol(path: Path) -> dict[str, float | int]:
    headers = [
        line
        for line in path.read_text(errors="replace").splitlines()
        if line.startswith("[box-cpal]")
    ]
    if len(headers) != 1:
        raise RuntimeError(f"{path}: expected one [box-cpal] protocol header")
    tokens = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", headers[0]))
    required = (
        "sigma_mode",
        "sigma_adapt",
        "al_sigma",
        "sigma_base",
        "lambda_param",
        "refl",
        "lambdaA",
    )
    missing = [name for name in required if name not in tokens]
    if missing:
        raise RuntimeError(f"{path}: protocol header is missing {missing}")

    sigma_mode = int(tokens["sigma_mode"])
    sigma_adapt = int(tokens["sigma_adapt"])
    al_sigma = float(tokens["al_sigma"])
    sigma_base = float(tokens["sigma_base"])
    lambda_param = float(tokens["lambda_param"])
    reflection = float(tokens["refl"])
    lambda_a = float(tokens["lambdaA"])
    if sigma_mode != 1 or sigma_adapt != 1:
        raise RuntimeError(
            f"{path}: expected sigma_mode=1 and sigma_adapt=1, found "
            f"{sigma_mode} and {sigma_adapt}"
        )
    if sigma_base <= 0.0 or lambda_param <= 0.0 or lambda_a <= 0.0:
        raise RuntimeError(f"{path}: nonpositive CP-AL metric scale")

    sigma_init_factor = al_sigma / sigma_base
    lambda_factor = lambda_param / lambda_a
    metric_normalization = sigma_base * math.sqrt(lambda_param)
    checks = (
        ("sigma initialization factor", sigma_init_factor, 0.5),
        ("lambda safety factor", lambda_factor, 2.01),
        ("metric normalization", metric_normalization, 1.0),
        ("reflection parameter", reflection, 1.0),
    )
    for label, actual, expected in checks:
        if not math.isclose(actual, expected, rel_tol=2e-3, abs_tol=2e-4):
            raise RuntimeError(
                f"{path}: {label} is {actual:.8g}, expected {expected:.8g}"
            )
    return {
        "sigma_mode": sigma_mode,
        "sigma_adapt": sigma_adapt,
        "sigma_init_factor_observed": sigma_init_factor,
        "lambda_factor_observed": lambda_factor,
        "reflection_observed": reflection,
    }


def finite_nonnegative(value: str, label: str, path: Path) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(f"{path}: invalid {label}={value}")
    return parsed


def scientific_latex(value: float) -> str:
    if value == 0.0:
        return r"\(0\)"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return rf"\({mantissa:.2f} \times 10^{{{exponent}}}\)"


def bold(value: str, enabled: bool) -> str:
    return rf"\textbf{{{value}}}" if enabled else value


def collect(manifest: Path, run_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    expected_manifest_sha256 = sha256(manifest)
    for task in read_manifest(manifest):
        for method in METHODS:
            method_dir = run_root / "runs" / task["instance"] / method
            summary = method_dir / "solver_output" / f"{task['instance']}_summary.txt"
            wrapper = method_dir / "wrapper.json"
            try:
                values = parse_summary(summary)
                protocol: dict[str, float | int | str] = {
                    "sigma_mode": "",
                    "sigma_adapt": "",
                    "sigma_init_factor_observed": "",
                    "lambda_factor_observed": "",
                    "reflection_observed": "",
                }
                if method == "rhr_cpal":
                    protocol.update(parse_cpal_protocol(method_dir / "solver.log"))
                wrapper_data = json.loads(wrapper.read_text(encoding="utf-8"))
                exit_code = int(wrapper_data["exit_code"])
                if exit_code != 0:
                    raise RuntimeError(f"{wrapper}: solver exit code {exit_code}")
                manifest_sha256 = recorded_sha256(method_dir / "manifest.sha256")
                if manifest_sha256 != expected_manifest_sha256:
                    raise RuntimeError(
                        f"{method_dir}: manifest SHA256 {manifest_sha256} does not match"
                    )
                binary_sha256 = recorded_sha256(method_dir / "binary.sha256")
                residuals = {
                    "rel_primal": finite_nonnegative(
                        values["Relative Primal Residual"], "relative primal residual", summary
                    ),
                    "rel_dual": finite_nonnegative(
                        values["Relative Dual Residual"], "relative dual residual", summary
                    ),
                    "rel_gap": finite_nonnegative(
                        values["Relative Objective Gap"], "relative objective gap", summary
                    ),
                }
                rows.append(
                    {
                        "family": task["family"],
                        "size": task["size"],
                        "instance": task["instance"],
                        "seed": int(task["seed"]),
                        "method": method,
                        "termination_reason": values["Termination Reason"],
                        "iterations": int(float(values["Iterations Count"])),
                        "runtime_sec": finite_nonnegative(
                            values["Runtime (sec)"], "runtime", summary
                        ),
                        "wrapper_wall_sec": finite_nonnegative(
                            str(wrapper_data["wall_seconds"]), "wrapper wall time", wrapper
                        ),
                        "exit_code": exit_code,
                        "manifest_sha256": manifest_sha256,
                        "binary_sha256": binary_sha256,
                        **protocol,
                        **residuals,
                        "score": max(residuals.values()),
                        "summary_sha256": sha256(summary),
                    }
                )
            except Exception as exc:  # audit all missing/corrupt rows together
                errors.append(f"{task['instance']}/{method}: {exc}")
    return rows, errors


def aggregate(rows: list[dict[str, object]], per_family: int) -> list[dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for method in METHODS:
            group = [row for row in rows if row["family"] == family and row["method"] == method]
            if len(group) != per_family:
                raise RuntimeError(
                    f"{family}/{method}: expected {per_family} rows, found {len(group)}"
                )
            scores = [float(row["score"]) for row in group]
            aggregate_rows.append(
                {
                    "family": family,
                    "method": method,
                    "instances": len(group),
                    "median_score": statistics.median(scores),
                    "score_le_1e-4": sum(value <= 1e-4 for value in scores),
                    "score_le_1e-6": sum(value <= 1e-6 for value in scores),
                    "optimal": sum(str(row["termination_reason"]).upper() == "OPTIMAL" for row in group),
                    "median_runtime_sec": statistics.median(float(row["runtime_sec"]) for row in group),
                }
            )
    return aggregate_rows


def latex_rows(aggregate_rows: list[dict[str, object]]) -> list[str]:
    lookup = {(row["family"], row["method"]): row for row in aggregate_rows}
    lines: list[str] = []
    for family in FAMILIES:
        left = lookup[(family, "cupdlpx")]
        right = lookup[(family, "rhr_cpal")]
        left_median = float(left["median_score"])
        right_median = float(right["median_score"])
        left_4, right_4 = int(left["score_le_1e-4"]), int(right["score_le_1e-4"])
        left_6, right_6 = int(left["score_le_1e-6"]), int(right["score_le_1e-6"])
        values = (
            bold(scientific_latex(left_median), left_median <= right_median),
            bold(scientific_latex(right_median), right_median <= left_median),
            bold(str(left_4), left_4 >= right_4),
            bold(str(right_4), right_4 >= left_4),
            bold(str(left_6), left_6 >= right_6),
            bold(str(right_6), right_6 >= left_6),
        )
        lines.append(f"{FAMILY_LABELS[family]:<14} & " + " & ".join(values) + r" \\")
    return lines


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def observed_range(rows: list[dict[str, object]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [min(values), max(values)] if values else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-per-family", type=int, default=1000)
    args = parser.parse_args()

    tasks = read_manifest(args.manifest)
    expected_tasks = len(FAMILIES) * args.expected_per_family
    if len(tasks) != expected_tasks:
        raise RuntimeError(f"expected {expected_tasks} manifest rows, found {len(tasks)}")

    rows, errors = collect(args.manifest, args.run_root)
    binary_hashes = {
        method: sorted(
            {
                str(row["binary_sha256"])
                for row in rows
                if row["method"] == method
            }
        )
        for method in METHODS
    }
    for method, hashes in binary_hashes.items():
        if len(hashes) != 1:
            errors.append(f"{method}: expected one binary SHA256, found {hashes}")
    cpal_protocol_rows = [row for row in rows if row["method"] == "rhr_cpal"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "collector_sha256": sha256(Path(__file__)),
        "runner_sha256": sha256(RUNNER_PATH),
        "manifest": args.manifest.name,
        "manifest_sha256": sha256(args.manifest),
        "score_definition": (
            "maximum of the solver-reported relative primal residual, "
            "relative dual residual, and relative objective gap"
        ),
        "residual_normalization": {
            "primal": "absolute primal residual / (1 + norm of constraint bounds)",
            "dual": "absolute dual residual / (1 + norm of objective vector)",
            "gap": "absolute primal-dual objective gap / (1 + |primal objective| + |dual objective|)",
            "shared_by_both_implementations": True,
            "independent_solution_recomputation": False,
        },
        "expected_instances": expected_tasks,
        "expected_run_rows": expected_tasks * len(METHODS),
        "collected_run_rows": len(rows),
        "binary_sha256_by_method": binary_hashes,
        "cpal_protocol": {
            "validated_rows": len(cpal_protocol_rows),
            "sigma_mode": sorted({int(row["sigma_mode"]) for row in cpal_protocol_rows}),
            "sigma_adapt": sorted({int(row["sigma_adapt"]) for row in cpal_protocol_rows}),
            "sigma_init_factor_range": observed_range(
                cpal_protocol_rows, "sigma_init_factor_observed"
            ),
            "lambda_factor_range": observed_range(
                cpal_protocol_rows, "lambda_factor_observed"
            ),
            "reflection": sorted(
                {float(row["reflection_observed"]) for row in cpal_protocol_rows}
            ),
            "unreliable_displacement_action": (
                "rollback to the best previously retained safeguarded sigma"
            ),
        },
        "errors": errors,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"random-LP audit found {len(errors)} errors; see audit.json")

    aggregates = aggregate(rows, args.expected_per_family)
    write_csv(args.output_dir / "random_lp_runs.csv", rows)
    write_csv(args.output_dir / "random_lp_table.csv", aggregates)
    (args.output_dir / "random_lp_table_rows.tex").write_text(
        "\n".join(latex_rows(aggregates)) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
