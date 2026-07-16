#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Audit all LP benchmark runs and generate MIPLIB/Mittelmann table cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path


BASELINE_IDS = ("cupdlpx_c", "cupdlpc", "cupdlp_jl", "hprlp_jl", "hprlp_c")
METHOD_ORDER = (*BASELINE_IDS, "rhr_cpal")
METHOD_LABELS = {
    "cupdlpx_c": "cuPDLPx(C)",
    "cupdlpc": "cuPDLP-C",
    "cupdlp_jl": "cuPDLP.jl",
    "hprlp_jl": "HPR-LP.jl",
    "hprlp_c": "HPR-LP-C",
    "rhr_cpal": "RHR-CP-AL",
}
LATEX_METHOD_LABELS = {
    key: rf"\texttt{{{label}}}" for key, label in METHOD_LABELS.items()
}
SPLITS = ("Small", "Medium", "Large")
TOLERANCES = ("1e-4", "1e-8")
SHIFT = 10.0
TIME_LIMIT_GRACE_SECONDS = 1.0
CPAL_ROUTES = {
    "lowobj_highcv_eval100_cr5",
    "zero_equality_cr0",
    "high_equality_tail_or_base",
}
INFRASTRUCTURE_PATTERNS = (
    "CUDA context cannot be initialized",
    "CUSPARSE_STATUS_NOT_INITIALIZED",
)

BASELINE_PARSER_PATH = Path(__file__).with_name("parse_outputs.py")
BASELINE_RUNNER_PATH = Path(__file__).with_name("run_baseline_group.sbatch")
CPAL_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "LP"
    / "CP_AL"
    / "run_unified_full856.sbatch"
)
_PARSER_SPEC = importlib.util.spec_from_file_location(
    "lp_baseline_output_parser", BASELINE_PARSER_PATH
)
if _PARSER_SPEC is None or _PARSER_SPEC.loader is None:
    raise RuntimeError(f"unable to load baseline parser: {BASELINE_PARSER_PATH}")
BASELINE_PARSER = importlib.util.module_from_spec(_PARSER_SPEC)
_PARSER_SPEC.loader.exec_module(BASELINE_PARSER)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_sha256(path: Path) -> str:
    token = path.read_text(encoding="utf-8").split(maxsplit=1)[0]
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise RuntimeError(f"{path}: invalid SHA256 record")
    return token


def validated_sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{label}: expected a lowercase SHA256 digest")
    return value


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 856:
        raise RuntimeError(f"expected 856 manifest rows, found {len(rows)}")
    identities = {(r["dataset"], r["split"], r["instance"], r["tolerance"]) for r in rows}
    if len(identities) != len(rows):
        raise RuntimeError("full856 manifest contains duplicate task identities")
    return rows


def read_baselines(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("**/summary_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source"] = str(path)
                rows.append(row)
    return rows


def reparse_baseline_raw_output(row: dict[str, str]) -> None:
    """Replace an intermediate CSV row by one parse of the retained raw output."""
    kind = row["solver_id"]
    instance_dir = Path(row["source"]).parent / "solver_outputs" / row["instance"]
    candidates = {
        "cupdlpx_c": instance_dir / f"{row['instance']}_summary.txt",
        "cupdlpc": instance_dir / "summary.json",
        "cupdlp_jl": instance_dir / f"{row['instance']}_summary.json",
        "hprlp_jl": instance_dir / "one.csv",
        "hprlp_c": instance_dir / "stdout.log",
    }
    parsers = {
        "cupdlpx_c": BASELINE_PARSER.parse_cupdlpx_txt,
        "cupdlpc": BASELINE_PARSER.parse_cupdlpc_json,
        "cupdlp_jl": BASELINE_PARSER.parse_cupdlp_jl_json,
        "hprlp_jl": BASELINE_PARSER.parse_hprlp_jl_csv,
        "hprlp_c": BASELINE_PARSER.parse_hprlp_c_text,
    }
    try:
        path = BASELINE_PARSER.resolve_solver_output(kind, candidates[kind])
        if not path.is_file():
            raise FileNotFoundError(f"missing raw solver output {path}")
        parsed = parsers[kind](path)
        row.update(parsed)
        row["error"] = ""
        row["source_output"] = str(path)
    except Exception as exc:
        row.update(
            {
                "status": "ERROR",
                "time_sec": "nan",
                "iterations": "0",
                "objective": "nan",
                "rel_gap": "nan",
                "rel_primal": "nan",
                "rel_dual": "nan",
                "error": str(exc).replace("\n", " "),
                "source_output": "",
            }
        )


def baseline_infrastructure_failure(row: dict[str, str]) -> str | None:
    source = Path(row["source"])
    instance_dir = source.parent / "solver_outputs" / row["instance"]
    for name in ("stdout.log", "stderr.log"):
        path = instance_dir / name
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for pattern in INFRASTRUCTURE_PATTERNS:
            if pattern in text:
                return f"{path}: {pattern}"
    return None


def parse_text_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    for key in ("Termination Reason", "Runtime (sec)"):
        if key not in values:
            raise RuntimeError(f"{path}: missing {key}")
    return values


def summary_value(values: dict[str, str], *keys: str) -> str:
    return next((values[key] for key in keys if key in values), "")


def parse_cpal_policy_log(path: Path) -> tuple[str, int]:
    text = path.read_text(errors="replace")
    routes = re.findall(r"\[unified-policy\][^\n]*\broute=([^\s]+)", text)
    if len(routes) != 1 or routes[0] not in CPAL_ROUTES:
        raise RuntimeError(f"{path}: expected one recognized unified-policy route")
    starts = text.count("[alpha-direct-pulse-start]")
    finals = re.findall(
        r"\[alpha-direct-pulse-final\][^\n]*\bpositive_charged_iters=(\d+)",
        text,
    )
    if starts > 1 or (starts == 1 and len(finals) != 1):
        raise RuntimeError(f"{path}: inconsistent positive-alpha pulse log")
    positive_iterations = int(finals[0]) if finals else 0
    if (starts == 1) != (positive_iterations > 0):
        raise RuntimeError(f"{path}: pulse start and charged iterations disagree")
    return routes[0], positive_iterations


def parse_wrapper(path: Path) -> tuple[float, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    wall_seconds = float(data["wall_seconds"])
    exit_code = int(data["exit_code"])
    if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise RuntimeError(f"{path}: invalid wrapper wall time")
    if exit_code != 0:
        raise RuntimeError(f"{path}: solver exit code {exit_code}")
    return wall_seconds, exit_code


def resolve_cpal_summary(task_dir: Path, instance: str) -> Path:
    stems = {instance, instance.split(".", 1)[0]}
    matches = [
        task_dir / "solver_output" / f"{stem}_summary.txt"
        for stem in sorted(stems)
        if (task_dir / "solver_output" / f"{stem}_summary.txt").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{task_dir}: expected one CP-AL summary for {instance}, found {matches}"
        )
    return matches[0]


def read_cpal(manifest_rows: list[dict[str, str]], root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    expected_manifest_sha256: str | None = None
    for index, task in enumerate(manifest_rows):
        matches = list(root.glob(f"task_{index:03d}_*"))
        if len(matches) != 1:
            raise RuntimeError(f"CP_AL task {index}: expected one directory, found {len(matches)}")
        summary = resolve_cpal_summary(matches[0], task["instance"])
        values = parse_text_summary(summary)
        route, positive_iterations = parse_cpal_policy_log(matches[0] / "solver.log")
        wrapper_wall_seconds, exit_code = parse_wrapper(matches[0] / "wrapper.json")
        manifest_sha256 = recorded_sha256(matches[0] / "manifest.sha256")
        if expected_manifest_sha256 is None:
            expected_manifest_sha256 = manifest_sha256
        elif manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(f"{matches[0]}: CP-AL manifest SHA256 changed within the run")
        binary_sha256 = recorded_sha256(matches[0] / "binary.sha256")
        rows.append(
            {
                "solver": METHOD_LABELS["rhr_cpal"],
                "solver_id": "rhr_cpal",
                "dataset": task["dataset"],
                "split": task["split"],
                "instance": task["instance"],
                "tolerance": task["tolerance"],
                "time_limit": task["time_limit"],
                "status": values["Termination Reason"],
                "time_sec": values["Runtime (sec)"],
                "wrapper_wall_sec": str(wrapper_wall_seconds),
                "exit_code": str(exit_code),
                "iterations": values.get("Iterations Count", ""),
                "objective": summary_value(
                    values, "Primal Objective", "Primal Objective Value"
                ),
                "rel_gap": values.get("Relative Objective Gap", ""),
                "rel_primal": values.get("Relative Primal Residual", ""),
                "rel_dual": values.get("Relative Dual Residual", ""),
                "error": "",
                "policy_route": route,
                "positive_alpha_iterations": str(positive_iterations),
                "manifest_sha256": manifest_sha256,
                "binary_sha256": binary_sha256,
                "source": str(summary),
            }
        )
    return rows


def status_ok(value: str) -> bool:
    status = (value or "").upper()
    return "OPTIMAL" in status and "INFEAS" not in status


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def residual_score(row: dict[str, object]) -> float | None:
    values = [finite_float(str(row.get(key, ""))) for key in ("rel_primal", "rel_dual", "rel_gap")]
    if any(value is None or value < 0.0 for value in values):
        return None
    return max(value for value in values if value is not None)


def quality_satisfied(row: dict[str, object]) -> bool:
    score = residual_score(row)
    tolerance = finite_float(str(row.get("tolerance", "")))
    return (
        status_ok(str(row.get("status", "")))
        and score is not None
        and tolerance is not None
        and score <= tolerance * (1.0 + 1e-8)
    )


def within_time_limit(row: dict[str, object]) -> bool:
    runtime = finite_float(str(row.get("time_sec", "")))
    limit = finite_float(str(row.get("time_limit", "")))
    return (
        runtime is not None
        and runtime >= 0.0
        and limit is not None
        and limit >= 0.0
        and runtime <= limit + TIME_LIMIT_GRACE_SECONDS
    )


def audited_solved(row: dict[str, object]) -> bool:
    return quality_satisfied(row) and within_time_limit(row)


def sgm10(times: list[float]) -> float:
    return math.exp(sum(math.log(value + SHIFT) for value in times) / len(times)) - SHIFT


def validate_rows(manifest_rows: list[dict[str, str]], rows: list[dict[str, str]]) -> list[str]:
    expected_tasks = {
        (r["dataset"], r["split"], r["instance"], r["tolerance"]): r for r in manifest_rows
    }
    errors: list[str] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        identity = (row["dataset"], row["split"], row["instance"], row["tolerance"])
        solver_id = row["solver_id"]
        full_identity = (solver_id, *identity)
        if solver_id not in METHOD_ORDER:
            errors.append(f"unknown solver id {solver_id}: {row.get('source', '')}")
        if identity not in expected_tasks:
            errors.append(f"unexpected task {identity}: {row.get('source', '')}")
        if full_identity in seen:
            errors.append(f"duplicate row {full_identity}")
        seen.add(full_identity)
    for solver_id in METHOD_ORDER:
        for identity in expected_tasks:
            if (solver_id, *identity) not in seen:
                errors.append(f"missing row {(solver_id, *identity)}")
    return errors


def summarize(rows: list[dict[str, str]], dataset: str, split: str | None, tolerance: str) -> dict[str, object]:
    group = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["tolerance"] == tolerance
        and (split is None or row["split"] == split)
    ]
    solved = 0
    charged: list[float] = []
    for row in group:
        limit = float(row["time_limit"])
        runtime = finite_float(row["time_sec"])
        if audited_solved(row) and runtime is not None:
            solved += 1
            charged.append(min(runtime, limit))
        else:
            charged.append(limit)
    return {"count": len(group), "solved": solved, "sgm10": sgm10(charged)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def portable_manifest(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "task_index": index,
            "dataset": row["dataset"],
            "split": row["split"],
            "instance": row["instance"],
            "tolerance": row["tolerance"],
            "time_limit": row["time_limit"],
            "input_file": Path(row["mps"]).name,
        }
        for index, row in enumerate(rows)
    ]


def input_hash_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    paths: dict[tuple[str, str, str], Path] = {}
    for row in rows:
        identity = (row["dataset"], row["split"], row["instance"])
        path = Path(row["mps"])
        previous = paths.setdefault(identity, path)
        if previous != path:
            raise RuntimeError(f"input path changes across tolerances: {identity}")
    output: list[dict[str, object]] = []
    for (dataset, split, instance), path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing benchmark input: {path}")
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "instance": instance,
                "input_file": path.name,
                "sha256": sha256(path),
            }
        )
    return output


def parse_software_provenance(path: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    hashes: list[dict[str, str]] = []
    revisions: list[dict[str, str]] = []
    metadata: list[str] = []
    for line in lines:
        hash_match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if hash_match:
            hashes.append(
                {"artifact": Path(hash_match.group(2)).name, "sha256": hash_match.group(1)}
            )
            continue
        if "\t" in line:
            source, revision = line.rsplit("\t", 1)
            if re.fullmatch(r"[0-9a-f]{40}|not-a-git-checkout", revision):
                revisions.append({"repository": Path(source).name, "revision": revision})
                continue
        metadata.append(line)
    return {
        "source_record_sha256": sha256(path),
        "metadata": metadata,
        "artifact_hashes": hashes,
        "source_revisions": revisions,
    }


def parse_execution_retries(path: Path) -> list[dict[str, str]]:
    required = {
        "job_id",
        "method",
        "task_scope",
        "task_indices",
        "cause",
        "disposition",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if fields != required:
        raise RuntimeError(f"{path}: unexpected retry-record fields {sorted(fields)}")
    seen_jobs: set[str] = set()
    for row in rows:
        if not row["job_id"].isdigit() or not row["task_indices"]:
            raise RuntimeError(f"{path}: malformed retry record {row}")
        if row["job_id"] in seen_jobs:
            raise RuntimeError(f"{path}: duplicate retry job {row['job_id']}")
        seen_jobs.add(row["job_id"])
        if row["method"] not in METHOD_ORDER:
            raise RuntimeError(f"{path}: unknown retry method {row['method']}")
        indices = row["task_indices"].split(",")
        if any(not token.isdigit() for token in indices) or len(indices) != len(
            set(indices)
        ):
            raise RuntimeError(f"{path}: malformed retry task indices")
        if not row["cause"].startswith("cuda_"):
            raise RuntimeError(f"{path}: non-infrastructure retry cause {row['cause']}")
        if row["disposition"] != "same manifest, binary, and solver settings":
            raise RuntimeError(f"{path}: retry configuration is not frozen")
    return rows


def released_rows(
    rows: list[dict[str, object]], baseline_root: Path, cpal_root: Path
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        root = cpal_root if row["solver_id"] == "rhr_cpal" else baseline_root
        for key in ("source", "source_output"):
            value = str(item.get(key, ""))
            if not value:
                continue
            try:
                item[key] = str(Path(value).relative_to(root))
            except ValueError:
                item[key] = Path(value).name
        item["error"] = str(item.get("error", "")).replace(
            str(root), f"<{row['solver_id']}-run>"
        )
        normalized.append(item)
    return normalized


def latex_count_time(
    row: dict[str, object], group: list[dict[str, object]]
) -> tuple[str, str]:
    best_count = max(int(item["solved"]) for item in group)
    best_time = min(
        float(item["sgm10"])
        for item in group
        if int(item["solved"]) == best_count
    )
    count = str(int(row["solved"]))
    time = f"{float(row['sgm10']):.3f}"
    if int(row["solved"]) == best_count:
        count = rf"\textbf{{{count}}}"
        if math.isclose(float(row["sgm10"]), best_time, rel_tol=0.0, abs_tol=5e-13):
            time = rf"\textbf{{{time}}}"
    return count, time


def miplib_latex_rows(rows: list[dict[str, object]]) -> list[str]:
    lookup = {
        (str(row["method_id"]), str(row["tolerance"]), str(row["split"])): row
        for row in rows
    }
    lines: list[str] = []
    for tolerance_index, tolerance in enumerate(TOLERANCES):
        if tolerance_index:
            lines.append(r"\midrule")
        groups = {
            split: [
                lookup[(method, tolerance, split)] for method in METHOD_ORDER
            ]
            for split in (*SPLITS, "Total")
        }
        for method_index, method in enumerate(METHOD_ORDER):
            prefix = rf"\(10^{{{int(math.log10(float(tolerance)))}}}\)" if method_index == 0 else ""
            cells: list[str] = []
            for split in (*SPLITS, "Total"):
                cells.extend(latex_count_time(lookup[(method, tolerance, split)], groups[split]))
            lines.append(
                f"{prefix}\n& {LATEX_METHOD_LABELS[method]}\n& "
                + " & ".join(cells)
                + r" \\"
            )
    return lines


def mittelmann_latex_rows(rows: list[dict[str, object]]) -> list[str]:
    lookup = {
        (str(row["method_id"]), str(row["tolerance"])): row for row in rows
    }
    groups = {
        tolerance: [lookup[(method, tolerance)] for method in METHOD_ORDER]
        for tolerance in TOLERANCES
    }
    lines: list[str] = []
    for method in METHOD_ORDER:
        cells: list[str] = []
        for tolerance in TOLERANCES:
            cells.extend(latex_count_time(lookup[(method, tolerance)], groups[tolerance]))
        lines.append(
            f"{LATEX_METHOD_LABELS[method]} & "
            + " & ".join(cells)
            + r" \\"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--cpal-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--software-provenance", type=Path)
    parser.add_argument("--execution-retries", type=Path)
    parser.add_argument("--formal-baseline-wrapper-sha256")
    parser.add_argument("--formal-cpal-wrapper-sha256")
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    baseline_rows = read_baselines(args.baseline_run_root)
    for row in baseline_rows:
        reparse_baseline_raw_output(row)
    cpal_rows = read_cpal(manifest_rows, args.cpal_run_root)
    expected_manifest_sha256 = sha256(args.manifest)
    if any(row["manifest_sha256"] != expected_manifest_sha256 for row in cpal_rows):
        raise RuntimeError("CP-AL task manifest SHA256 does not match the collector manifest")
    rows = baseline_rows + cpal_rows
    for row in rows:
        row.setdefault("policy_route", "")
        row.setdefault("positive_alpha_iterations", "")
        row.setdefault("manifest_sha256", "")
        row.setdefault("binary_sha256", "")
        row.setdefault("source_output", "")
        row.setdefault("wrapper_wall_sec", "")
        row.setdefault("exit_code", "")
    errors = validate_rows(manifest_rows, rows)
    infrastructure_failures = [
        failure
        for row in baseline_rows
        if (failure := baseline_infrastructure_failure(row)) is not None
    ]
    errors.extend(f"unresolved infrastructure failure: {item}" for item in infrastructure_failures)
    retry_rows: list[dict[str, str]] = []
    if args.execution_retries is not None:
        try:
            retry_rows = parse_execution_retries(args.execution_retries)
        except Exception as exc:
            errors.append(str(exc))
    cpal_binary_hashes = sorted({row["binary_sha256"] for row in cpal_rows})
    if len(cpal_binary_hashes) != 1:
        errors.append(f"expected one CP-AL binary SHA256, found {cpal_binary_hashes}")

    status_residual_disagreements: list[dict[str, object]] = []
    over_time_optimal_rows: list[dict[str, object]] = []
    for row in rows:
        score = residual_score(row)
        runtime = finite_float(row["time_sec"])
        time_limit = finite_float(row["time_limit"])
        quality_ok = quality_satisfied(row)
        row["reported_optimal"] = status_ok(row["status"])
        row["audited_score"] = score if score is not None else ""
        row["audited_solved"] = audited_solved(row)
        if status_ok(row["status"]) and score is None:
            errors.append(
                f"optimal row has missing/nonfinite relative residual: "
                f"{row['solver_id']}/{row['dataset']}/{row['split']}/"
                f"{row['instance']}/{row['tolerance']}"
            )
        elif status_ok(row["status"]) and not quality_ok:
            status_residual_disagreements.append(
                {
                    "solver_id": row["solver_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "instance": row["instance"],
                    "tolerance": row["tolerance"],
                    "audited_score": score,
                }
            )
        if quality_ok and (
            runtime is None
            or runtime < 0.0
            or time_limit is None
            or time_limit < 0.0
        ):
            errors.append(
                f"optimal within-tolerance row has invalid runtime or time limit: "
                f"{row['solver_id']}/{row['dataset']}/{row['split']}/"
                f"{row['instance']}/{row['tolerance']}"
            )
        elif quality_ok and not within_time_limit(row):
            over_time_optimal_rows.append(
                {
                    "solver_id": row["solver_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "instance": row["instance"],
                    "tolerance": row["tolerance"],
                    "runtime": runtime,
                    "time_limit": time_limit,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashed_inputs = input_hash_rows(manifest_rows)
    if len(hashed_inputs) != 428:
        raise RuntimeError(f"expected 428 unique benchmark inputs, found {len(hashed_inputs)}")
    input_hash_path = args.output_dir / "lp_input_hashes.tsv"
    write_tsv(input_hash_path, hashed_inputs)
    audit = {
        "collector_sha256": sha256(Path(__file__)),
        "baseline_parser_sha256": sha256(BASELINE_PARSER_PATH),
        "baseline_raw_outputs_reparsed": True,
        "released_baseline_wrapper_sha256": sha256(BASELINE_RUNNER_PATH),
        "released_cpal_wrapper_sha256": sha256(CPAL_RUNNER_PATH),
        "formal_baseline_wrapper_sha256": validated_sha256(
            args.formal_baseline_wrapper_sha256, "formal baseline wrapper SHA256"
        )
        if args.formal_baseline_wrapper_sha256
        else None,
        "formal_cpal_wrapper_sha256": validated_sha256(
            args.formal_cpal_wrapper_sha256, "formal CP-AL wrapper SHA256"
        )
        if args.formal_cpal_wrapper_sha256
        else None,
        "manifest_sha256": expected_manifest_sha256,
        "manifest": args.manifest.name,
        "unique_input_files": len(hashed_inputs),
        "input_hash_file_sha256": sha256(input_hash_path),
        "expected_rows": len(manifest_rows) * len(METHOD_ORDER),
        "runtime_definition": "solver-native time_sec parsed from each method summary",
        "runtime_limit_grace_seconds": TIME_LIMIT_GRACE_SECONDS,
        "sgm_shift_seconds": SHIFT,
        "baseline_rows": len(baseline_rows),
        "cpal_rows": len(cpal_rows),
        "cpal_binary_sha256": cpal_binary_hashes,
        "cpal_policy_routes": dict(
            sorted(Counter(row["policy_route"] for row in cpal_rows).items())
        ),
        "cpal_positive_alpha_tasks": sum(
            int(row["positive_alpha_iterations"]) > 0 for row in cpal_rows
        ),
        "cpal_positive_alpha_iterations": sum(
            int(row["positive_alpha_iterations"]) for row in cpal_rows
        ),
        "termination_status_counts": {
            f"{solver_id}/{status}": count
            for (solver_id, status), count in sorted(
                Counter((row["solver_id"], row["status"]) for row in rows).items()
            )
        },
        "solver_error_rows": [
            {
                "solver_id": row["solver_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "instance": row["instance"],
                "tolerance": row["tolerance"],
                "error": str(row["error"]).replace(
                    str(args.baseline_run_root), "<baseline-run>"
                ),
            }
            for row in rows
            if row["status"] == "ERROR"
        ],
        "unresolved_infrastructure_failures": infrastructure_failures,
        "execution_retries_sha256": (
            sha256(args.execution_retries)
            if args.execution_retries is not None and args.execution_retries.is_file()
            else None
        ),
        "execution_retry_records": len(retry_rows),
        "reported_optimal_but_above_tolerance": status_residual_disagreements,
        "optimal_within_tolerance_but_over_time_limit": over_time_optimal_rows,
        "errors": errors,
    }
    provenance = None
    if args.software_provenance is not None:
        if not args.software_provenance.is_file():
            errors.append(f"missing software provenance: {args.software_provenance}")
        else:
            provenance = parse_software_provenance(args.software_provenance)
            audit["software_provenance_sha256"] = provenance["source_record_sha256"]
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"LP benchmark audit found {len(errors)} errors; see audit.json")

    miplib: list[dict[str, object]] = []
    mittelmann: list[dict[str, object]] = []
    for solver_id in METHOD_ORDER:
        solver_rows = [row for row in rows if row["solver_id"] == solver_id]
        for tolerance in TOLERANCES:
            for split in (*SPLITS, None):
                cell = summarize(solver_rows, "MIPLIB", split, tolerance)
                miplib.append(
                    {
                        "method_id": solver_id,
                        "method": METHOD_LABELS[solver_id],
                        "tolerance": tolerance,
                        "split": split or "Total",
                        **cell,
                    }
                )
            cell = summarize(solver_rows, "MITTELMANN", None, tolerance)
            mittelmann.append(
                {
                    "method_id": solver_id,
                    "method": METHOD_LABELS[solver_id],
                    "tolerance": tolerance,
                    **cell,
                }
            )

    write_csv(
        args.output_dir / "lp_all_runs.csv",
        released_rows(rows, args.baseline_run_root, args.cpal_run_root),
    )
    write_csv(args.output_dir / "lp_miplib_table.csv", miplib)
    write_csv(args.output_dir / "lp_mittelmann_table.csv", mittelmann)
    write_tsv(args.output_dir / "lp_task_manifest.tsv", portable_manifest(manifest_rows))
    if provenance is not None:
        (args.output_dir / "lp_software_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    (args.output_dir / "lp_miplib_table_rows.tex").write_text(
        "\n".join(miplib_latex_rows(miplib)) + "\n", encoding="utf-8"
    )
    (args.output_dir / "lp_mittelmann_table_rows.tex").write_text(
        "\n".join(mittelmann_latex_rows(mittelmann)) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
