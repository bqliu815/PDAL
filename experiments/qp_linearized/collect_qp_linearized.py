#!/usr/bin/env python3
"""Build the canonical five-method planted-KKT QP table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = (
    ("pdhcg", "PDHCG-II"),
    ("cpal", "RHR-CP-AL"),
    ("facp", "RHR-FA-CP"),
    ("lin_pdhg", "RHR-Lin-PDHG"),
    ("lin_cpal", "RHR-Lin-CP-AL"),
)
EXACT_KEYS = {"pdhcg", "cpal", "facp"}
FRESH_PROFILES = {
    "pdhcg": "pdhcg_original",
    "cpal": "rhr_cpal_subproblem",
    "facp": "rhr_facp_subproblem",
    "lin_pdhg": "rhr_lin_pdhg",
    "lin_cpal": "rhr_lin_cpal",
}
NATIVE_RUNTIME_PATTERN = re.compile(
    r"^Runtime \(sec\):\s*([^\s]+)\s*$", re.MULTILINE
)
RUN_FIELDS = (
    "method_key",
    "method",
    "source_run",
    "profile",
    "instance",
    "family_key",
    "h_family",
    "box_regime",
    "size_class",
    "elapsed_sec",
    "wrapper_elapsed_sec",
    "total_iters",
    "rel_primal",
    "rel_dual",
    "rel_gap",
    "status",
    "optimal",
    "termination_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-root",
        type=Path,
        help="fresh five-profile results directory written by run_qp_linearized.sbatch",
    )
    source.add_argument(
        "--exact-joined",
        type=Path,
        help="joined.csv from the original five-method run",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="generated manifest.csv; required with --run-root",
    )
    parser.add_argument(
        "--exact-source-run",
        help="exact-map run label (inferred from the path if omitted)",
    )
    parser.add_argument(
        "--lin-pdhg-root",
        type=Path,
        help="directory containing chunk_*/summary.csv for Lin-PDHG",
    )
    parser.add_argument(
        "--lin-cpal-root",
        type=Path,
        help="directory containing chunk_*/summary.csv for Lin-CP-AL",
    )
    parser.add_argument(
        "--lin-pdhg-source-run",
        help="Lin-PDHG run label (inferred from the path if omitted)",
    )
    parser.add_argument(
        "--lin-cpal-source-run",
        help="Lin-CP-AL run label (inferred from the path if omitted)",
    )
    parser.add_argument(
        "--exact-native-root",
        type=Path,
        help=(
            "five-method run root containing native *_summary.txt files; "
            "when supplied, native runtimes are required for every method"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_id_from_joined(path: Path) -> str:
    if path.parent.name == "analysis":
        return path.parent.parent.name
    return path.parent.name


def run_id_from_profile(path: Path) -> str:
    resolved = path.resolve()
    if resolved.parent.name == "results":
        return resolved.parent.parent.name
    return resolved.parent.name


def parse_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: nonfinite {field}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_tree_sha256(files: list[Path], root: Path) -> str:
    digest_lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())
    ]
    return hashlib.sha256("".join(digest_lines).encode()).hexdigest()


def native_runtime_index(
    files: list[Path], *, root: Path, context: str
) -> tuple[dict[str, float], dict[str, object]]:
    if not files:
        raise ValueError(f"{context}: no native summary files")
    runtimes: dict[str, float] = {}
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        instance = path.name.removesuffix("_summary.txt")
        if instance in runtimes:
            raise ValueError(f"{context}: duplicate native summary for {instance}")
        contents = path.read_text(encoding="utf-8")
        match = NATIVE_RUNTIME_PATTERN.search(contents)
        if match is None:
            raise ValueError(f"{context}/{instance}: missing Runtime (sec)")
        try:
            runtime = float(match.group(1))
        except ValueError as exc:
            raise ValueError(f"{context}/{instance}: invalid Runtime (sec)") from exc
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError(f"{context}/{instance}: invalid Runtime (sec)")
        runtimes[instance] = runtime
    return runtimes, {
        "summary_count": len(files),
        "tree_sha256": file_tree_sha256(files, root),
    }


def apply_native_runtimes(
    rows: list[dict[str, object]], runtimes: dict[str, float], *, context: str
) -> None:
    instances = {str(row["instance"]) for row in rows}
    missing = sorted(instances - runtimes.keys())
    extra = sorted(runtimes.keys() - instances)
    if missing or extra:
        raise ValueError(
            f"{context}: native-summary instance mismatch "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    for row in rows:
        row["elapsed_sec"] = runtimes[str(row["instance"])]


def normalize_row(
    row: dict[str, str],
    *,
    method_key: str,
    method: str,
    source_run: str,
    profile: str,
    metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    instance = row.get("instance") or row.get("instance_name")
    if not instance:
        raise ValueError(f"{method}: row without an instance")
    meta = metadata.get(instance, row)
    context = f"{method}/{instance}"
    wrapper_elapsed = parse_float(row, "elapsed_sec", context)
    return {
        "method_key": method_key,
        "method": method,
        "source_run": source_run,
        "profile": profile,
        "instance": instance,
        "family_key": meta.get("family_key", ""),
        "h_family": meta.get("h_family", ""),
        "box_regime": meta.get("box_regime", ""),
        "size_class": meta.get("size_class", ""),
        "elapsed_sec": wrapper_elapsed,
        "wrapper_elapsed_sec": wrapper_elapsed,
        "total_iters": int(parse_float(row, "total_iters", context)),
        "rel_primal": parse_float(row, "rel_primal", context),
        "rel_dual": parse_float(row, "rel_dual", context),
        "rel_gap": parse_float(row, "rel_gap", context),
        "status": row.get("status", ""),
        "optimal": row.get("optimal", ""),
        "termination_reason": row.get("termination_reason", ""),
    }


def load_profile(
    root: Path,
    *,
    method_key: str,
    method: str,
    metadata: dict[str, dict[str, str]],
    source_run: str | None = None,
) -> list[dict[str, object]]:
    files = sorted(root.glob("chunk_*/summary.csv"))
    if not files:
        raise ValueError(f"no chunk summaries below {root}")
    source_run = source_run or run_id_from_profile(root)
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(
            normalize_row(
                row,
                method_key=method_key,
                method=method,
                source_run=source_run,
                profile=root.name,
                metadata=metadata,
            )
            for row in read_rows(path)
        )
    return rows


def validate_method(
    rows: list[dict[str, object]], expected_count: int, tolerance: float
) -> set[str]:
    if len(rows) != expected_count:
        method = rows[0]["method"] if rows else "unknown"
        raise ValueError(f"{method}: expected {expected_count} rows, found {len(rows)}")
    instances = [str(row["instance"]) for row in rows]
    if len(set(instances)) != len(instances):
        raise ValueError(f"{rows[0]['method']}: duplicate instances")
    for row in rows:
        context = f"{row['method']}/{row['instance']}"
        if row["status"] != "ok" or row["termination_reason"] != "OPTIMAL":
            raise ValueError(f"{context}: run is not OPTIMAL")
        if str(row["optimal"]) not in {"1", "True", "true"}:
            raise ValueError(f"{context}: optimal flag is false")
        if (
            float(row["elapsed_sec"]) < 0
            or float(row["wrapper_elapsed_sec"]) < 0
            or int(row["total_iters"]) <= 0
        ):
            raise ValueError(f"{context}: invalid elapsed time or iteration count")
        residual = max(
            float(row["rel_primal"]),
            float(row["rel_dual"]),
            float(row["rel_gap"]),
        )
        if residual > tolerance * (1.0 + 1e-10):
            raise ValueError(
                f"{context}: residual {residual:.6g} exceeds {tolerance:.6g}"
            )
    return set(instances)


def sgm10(values: list[float]) -> float:
    return math.exp(sum(math.log(value + 10.0) for value in values) / len(values)) - 10.0


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    elapsed = [float(row["wrapper_elapsed_sec"]) for row in rows]
    per_iteration = [
        float(row["wrapper_elapsed_sec"]) / int(row["total_iters"]) * 1e4
        for row in rows
    ]
    first = rows[0]
    return {
        "method_key": first["method_key"],
        "method": first["method"],
        "total": len(rows),
        "solved": len(rows),
        "unsolved": 0,
        "total_elapsed": sum(elapsed),
        "median_elapsed": statistics.median(elapsed),
        "sgm10": sgm10(elapsed),
        "median_sec_per_1e4_iters": statistics.median(per_iteration),
        "source_run": first["source_run"],
        "profile": first["profile"],
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    display = dict(METHODS)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    native_provenance: dict[str, dict[str, object]] = {}
    profile_roots: dict[str, Path] = {}
    fresh_mode = args.run_root is not None

    if fresh_mode:
        if args.manifest is None:
            raise ValueError("--manifest is required with --run-root")
        for row in read_rows(args.manifest):
            instance = row.get("instance") or row.get("instance_name")
            if instance:
                metadata[instance] = row
        if not metadata:
            raise ValueError(f"manifest has no instance rows: {args.manifest}")
        fresh_source = run_id_from_profile(args.run_root)
        for key, _ in METHODS:
            root = args.run_root / FRESH_PROFILES[key]
            profile_roots[key] = root
            grouped[key] = load_profile(
                root,
                method_key=key,
                method=display[key],
                metadata=metadata,
                source_run=fresh_source,
            )
            files = list(root.glob("chunk_*/solver_outputs/*/*_summary.txt"))
            runtimes, provenance = native_runtime_index(
                files,
                root=root,
                context=f"native {display[key]}",
            )
            apply_native_runtimes(
                grouped[key], runtimes, context=f"native {display[key]}"
            )
            provenance["source_run"] = fresh_source
            native_provenance[key] = provenance
    else:
        if args.lin_pdhg_root is None or args.lin_cpal_root is None:
            raise ValueError(
                "--lin-pdhg-root and --lin-cpal-root are required with --exact-joined"
            )
        exact_input = read_rows(args.exact_joined)
        for row in exact_input:
            instance = row.get("instance") or row.get("instance_name")
            if instance:
                metadata.setdefault(instance, row)
        exact_source = args.exact_source_run or run_id_from_joined(args.exact_joined)
        for row in exact_input:
            key = row.get("method_key", "")
            if key in EXACT_KEYS:
                grouped[key].append(
                    normalize_row(
                        row,
                        method_key=key,
                        method=display[key],
                        source_run=exact_source,
                        profile="original fixed profile",
                        metadata=metadata,
                    )
                )
        grouped["lin_pdhg"] = load_profile(
            args.lin_pdhg_root,
            method_key="lin_pdhg",
            method=display["lin_pdhg"],
            metadata=metadata,
            source_run=args.lin_pdhg_source_run,
        )
        grouped["lin_cpal"] = load_profile(
            args.lin_cpal_root,
            method_key="lin_cpal",
            method=display["lin_cpal"],
            metadata=metadata,
            source_run=args.lin_cpal_source_run,
        )
        profile_roots = {
            "lin_pdhg": args.lin_pdhg_root,
            "lin_cpal": args.lin_cpal_root,
        }

        if args.exact_native_root is not None:
            for key in sorted(EXACT_KEYS):
                files = list(
                    args.exact_native_root.glob(
                        f"*/{key}/default/solver_outputs/*/*_summary.txt"
                    )
                )
                runtimes, provenance = native_runtime_index(
                    files,
                    root=args.exact_native_root,
                    context=f"native {display[key]}",
                )
                apply_native_runtimes(
                    grouped[key], runtimes, context=f"native {display[key]}"
                )
                provenance["source_run"] = exact_source
                native_provenance[key] = provenance
            for key, root in profile_roots.items():
                files = list(root.glob("chunk_*/solver_outputs/*/*_summary.txt"))
                runtimes, provenance = native_runtime_index(
                    files,
                    root=root,
                    context=f"native {display[key]}",
                )
                apply_native_runtimes(
                    grouped[key], runtimes, context=f"native {display[key]}"
                )
                provenance["source_run"] = grouped[key][0]["source_run"]
                native_provenance[key] = provenance

    instance_sets = {
        key: validate_method(rows, args.expected_count, args.tolerance)
        for key, rows in grouped.items()
    }
    reference = instance_sets["pdhcg"]
    mismatches = [key for key, values in instance_sets.items() if values != reference]
    if mismatches:
        raise ValueError(f"method instance sets differ: {', '.join(mismatches)}")

    ordered_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    for key, _ in METHODS:
        rows = sorted(grouped[key], key=lambda row: str(row["instance"]))
        ordered_rows.extend(rows)
        aggregate_rows.append(aggregate(rows))
        by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_family[str(row["family_key"])].append(row)
        for family in sorted(by_family):
            summary = aggregate(by_family[family])
            summary["family_key"] = family
            family_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", ordered_rows, list(RUN_FIELDS))
    aggregate_fields = list(aggregate_rows[0])
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows, aggregate_fields)
    write_csv(
        args.output_dir / "by_family.csv",
        family_rows,
        ["family_key", *aggregate_fields],
    )
    elapsed_definition = (
        "per-invocation wall time recorded by the experiment wrapper; "
        "scheduler queue time excluded"
    )
    timing_source = "per_invocation_wrapper"

    if fresh_mode:
        profile_artifacts: dict[str, dict[str, object]] = {}
        for key, root in profile_roots.items():
            csvs = sorted(root.glob("chunk_*/summary.csv"))
            profile_artifacts[key] = {
                "count": len(csvs),
                "tree_sha256": file_tree_sha256(csvs, root),
            }
        input_artifacts = {
            "collector_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(args.manifest),
            "profile_chunk_summaries": profile_artifacts,
        }
    else:
        lin_pdhg_csvs = sorted(args.lin_pdhg_root.glob("chunk_*/summary.csv"))
        lin_cpal_csvs = sorted(args.lin_cpal_root.glob("chunk_*/summary.csv"))
        input_artifacts = {
            "collector_sha256": sha256(Path(__file__)),
            "exact_joined_sha256": sha256(args.exact_joined),
            "lin_pdhg_chunk_summaries": {
                "count": len(lin_pdhg_csvs),
                "tree_sha256": file_tree_sha256(lin_pdhg_csvs, args.lin_pdhg_root),
            },
            "lin_cpal_chunk_summaries": {
                "count": len(lin_cpal_csvs),
                "tree_sha256": file_tree_sha256(lin_cpal_csvs, args.lin_cpal_root),
            },
        }

    audit = {
        "expected_instances_per_method": args.expected_count,
        "tolerance": args.tolerance,
        "method_instance_sets_identical": True,
        "all_rows_optimal_and_within_tolerance": True,
        "timing_source": timing_source,
        "elapsed_time_definition": elapsed_definition,
        "display_elapsed_field": "wrapper_elapsed_sec",
        "wrapper_elapsed_time_definition": (
            "per-invocation wall time recorded by the experiment wrapper; "
            "scheduler queue time excluded"
        ),
        "native_runtime_time_definition": (
            "when native summaries are supplied, elapsed_sec stores the "
            "solver-reported Runtime (sec); it is retained for audit and is "
            "not used in the displayed aggregate"
        ),
        "sgm_shift_seconds": 10.0,
        "input_artifacts": input_artifacts,
        "native_runtime_summaries": native_provenance,
        "methods": aggregate_rows,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
