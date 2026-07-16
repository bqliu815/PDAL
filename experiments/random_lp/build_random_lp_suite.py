#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Build a reproducible synthetic equality-LP suite from fixed family standards.

The generated problems have the form

    min c^T x
    s.t. A x = b
         l <= x <= u

Families are defined by two difficulty axes:

1. Scaling severity: controlled by row/column log10 scaling ranges.
2. Near-dependence severity: controlled by the fraction of overwritten rows/cols
   and the correlation strength used in the overwrite.

For reproducibility, all family definitions and size buckets live in this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


FAMILY_SPECS: dict[str, dict[str, object]] = {
    "easy": {
        "description": "Low scaling and low near-dependence.",
        "row_scale_log10": 2.0,
        "col_scale_log10": 2.0,
        "near_dep_col_fraction": 0.02,
        "near_dep_row_fraction": 0.02,
        "correlation_strength": 0.98,
    },
    "scaled": {
        "description": "High scaling and low near-dependence.",
        "row_scale_log10": 5.0,
        "col_scale_log10": 5.0,
        "near_dep_col_fraction": 0.02,
        "near_dep_row_fraction": 0.02,
        "correlation_strength": 0.98,
    },
    "neardep": {
        "description": "Low scaling and high near-dependence.",
        "row_scale_log10": 2.0,
        "col_scale_log10": 2.0,
        "near_dep_col_fraction": 0.28,
        "near_dep_row_fraction": 0.28,
        "correlation_strength": 0.9995,
    },
    "combo": {
        "description": "Moderate scaling and moderate near-dependence.",
        "row_scale_log10": 3.0,
        "col_scale_log10": 3.0,
        "near_dep_col_fraction": 0.18,
        "near_dep_row_fraction": 0.18,
        "correlation_strength": 0.995,
    },
}


SIZE_SPECS: dict[str, dict[str, float | int]] = {
    "s": {"m": 150, "n": 400, "density": 0.045},
    "m": {"m": 250, "n": 700, "density": 0.025},
    "l": {"m": 400, "n": 1000, "density": 0.012},
}


SIZE_ORDER = ("s", "m", "l")
DEFAULT_SIZE_WEIGHTS = (0.4, 0.4, 0.2)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build synthetic equality-LP suites.")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for the generated suite.",
    )
    parser.add_argument(
        "--per_family",
        type=int,
        default=1000,
        help="Number of instances per family.",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=2026042401,
        help="Base RNG seed for the whole suite.",
    )
    parser.add_argument(
        "--generator",
        default=str(here / "generate_pure_equality_lp.py"),
        help="Path to generate_pure_equality_lp.py.",
    )
    parser.add_argument(
        "--small_weight",
        type=float,
        default=DEFAULT_SIZE_WEIGHTS[0],
        help="Fraction of small instances within each family.",
    )
    parser.add_argument(
        "--medium_weight",
        type=float,
        default=DEFAULT_SIZE_WEIGHTS[1],
        help="Fraction of medium instances within each family.",
    )
    parser.add_argument(
        "--large_weight",
        type=float,
        default=DEFAULT_SIZE_WEIGHTS[2],
        help="Fraction of large instances within each family.",
    )
    parser.add_argument(
        "--bound_scale_log10",
        type=float,
        default=3.0,
        help="Shared box-width scaling parameter.",
    )
    parser.add_argument(
        "--objective_noise",
        type=float,
        default=1e-6,
        help="Shared objective noise level.",
    )
    parser.add_argument(
        "--bound_active_fraction",
        type=float,
        default=0.30,
        help="Shared planted active-bound fraction.",
    )
    return parser.parse_args()


def allocate_counts(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight for weight in weights]
    base = [math.floor(value) for value in raw]
    remainder = total - sum(base)
    fractional = sorted(
        ((raw[idx] - base[idx], idx) for idx in range(len(weights))),
        reverse=True,
    )
    for _, idx in fractional[:remainder]:
        base[idx] += 1
    return base


def run_batch(
    generator: str,
    out_dir: Path,
    prefix: str,
    count: int,
    seed: int,
    batch_params: dict[str, object],
) -> list[dict[str, str]]:
    if count <= 0:
        return []

    cmd = [
        sys.executable,
        generator,
        "--out_dir",
        str(out_dir),
        "--name",
        prefix,
        "--count",
        str(count),
        "--seed",
        str(seed),
    ]
    for key, value in batch_params.items():
        cmd.extend([f"--{key}", str(value)])

    subprocess.run(cmd, check=True)

    manifest_path = out_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open()))
    batch_manifest_path = out_dir / f"manifest_{prefix}.csv"
    with batch_manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest_path.unlink()
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("manifest*.csv", "instances.txt", "suite_spec.json"):
        for path in out_dir.glob(pattern):
            path.unlink()

    weights = [args.small_weight, args.medium_weight, args.large_weight]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("size weights must sum to a positive number")
    weights = [weight / weight_sum for weight in weights]
    counts = allocate_counts(args.per_family, weights)

    suite_spec = {
        "per_family": args.per_family,
        "seed_base": args.seed_base,
        "size_weights": dict(zip(SIZE_ORDER, weights)),
        "size_counts": dict(zip(SIZE_ORDER, counts)),
        "families": FAMILY_SPECS,
        "sizes": SIZE_SPECS,
        "shared": {
            "bound_scale_log10": args.bound_scale_log10,
            "objective_noise": args.objective_noise,
            "bound_active_fraction": args.bound_active_fraction,
        },
    }
    (out_dir / "suite_spec.json").write_text(
        json.dumps(suite_spec, indent=2),
        encoding="utf-8",
    )

    all_rows: list[dict[str, str]] = []
    instance_names: list[str] = []
    next_seed = args.seed_base

    for family, family_params in FAMILY_SPECS.items():
        for size_label, size_count in zip(SIZE_ORDER, counts):
            prefix = f"{family}_{size_label}"
            batch_params = {
                **SIZE_SPECS[size_label],
                **{k: v for k, v in family_params.items() if k != "description"},
                "bound_scale_log10": args.bound_scale_log10,
                "objective_noise": args.objective_noise,
                "bound_active_fraction": args.bound_active_fraction,
            }
            batch_rows = run_batch(
                generator=args.generator,
                out_dir=out_dir,
                prefix=prefix,
                count=size_count,
                seed=next_seed,
                batch_params=batch_params,
            )
            next_seed += size_count
            all_rows.extend(batch_rows)
            instance_names.extend(row["instance"] for row in batch_rows)

    if all_rows:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        task_fields = (
            "family",
            "size",
            "instance",
            "seed",
            "m",
            "n",
            "nnz",
            "density_realized",
            "mps",
        )
        task_rows = []
        for row in all_rows:
            tokens = row["instance"].split("_")
            task_rows.append(
                {
                    "family": tokens[0],
                    "size": tokens[1],
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "m": row["m"],
                    "n": row["n"],
                    "nnz": row["nnz"],
                    "density_realized": row["density_realized"],
                    "mps": f"{row['instance']}.mps",
                }
            )
        task_manifest = out_dir / "suite_manifest.tsv"
        with task_manifest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=task_fields,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(task_rows)

    with (out_dir / "instances.txt").open("w", encoding="utf-8") as fh:
        for name in instance_names:
            fh.write(f"{name}\n")

    print(f"Built suite in {out_dir}")
    print(f"Per family: {args.per_family}")
    print(f"Size counts: {dict(zip(SIZE_ORDER, counts))}")
    print(f"Instances: {len(instance_names)}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    if all_rows:
        digest = hashlib.sha256((out_dir / "suite_manifest.tsv").read_bytes()).hexdigest()
        print(f"Task manifest: {out_dir / 'suite_manifest.tsv'}")
        print(f"Task manifest SHA256: {digest}")


if __name__ == "__main__":
    main()
