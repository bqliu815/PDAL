#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Audit an existing generated suite and write the portable task manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


FAMILIES = ("easy", "scaled", "neardep", "combo")
SIZES = ("s", "m", "l")
FIELDS = (
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
HASH_FIELDS = ("instance", "mps", "sha256")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hash-output",
        type=Path,
        help="optional TSV of SHA256 values for every generated MPS file",
    )
    parser.add_argument("--per-family", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=2026042401)
    args = parser.parse_args()

    source = args.suite_dir / "manifest.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_total = len(FAMILIES) * args.per_family
    if len(rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} source rows, found {len(rows)}")

    task_rows: list[dict[str, str]] = []
    identities: set[str] = set()
    for index, row in enumerate(rows):
        tokens = row["instance"].split("_", 2)
        if len(tokens) < 2 or tokens[0] not in FAMILIES or tokens[1] not in SIZES:
            raise RuntimeError(f"invalid instance identity: {row['instance']}")
        if row["instance"] in identities:
            raise RuntimeError(f"duplicate instance: {row['instance']}")
        identities.add(row["instance"])
        expected_seed = args.seed_base + index
        if int(row["seed"]) != expected_seed:
            raise RuntimeError(
                f"{row['instance']}: expected seed {expected_seed}, found {row['seed']}"
            )
        mps_name = f"{row['instance']}.mps"
        if not (args.suite_dir / mps_name).is_file():
            raise RuntimeError(f"missing MPS file: {args.suite_dir / mps_name}")
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
                "mps": mps_name,
            }
        )

    family_counts = Counter(row["family"] for row in task_rows)
    expected_families = Counter({family: args.per_family for family in FAMILIES})
    if family_counts != expected_families:
        raise RuntimeError(f"family count mismatch: {family_counts}")
    if args.per_family == 1000:
        expected_sizes = Counter(
            {
                (family, size): count
                for family in FAMILIES
                for size, count in zip(SIZES, (400, 400, 200), strict=True)
            }
        )
        actual_sizes = Counter((row["family"], row["size"]) for row in task_rows)
        if actual_sizes != expected_sizes:
            raise RuntimeError(f"size count mismatch: {actual_sizes}")

    output = args.output or args.suite_dir / "suite_manifest.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)
    print(f"rows={len(task_rows)} sha256={sha256(output)} output={output}")

    if args.hash_output:
        args.hash_output.parent.mkdir(parents=True, exist_ok=True)
        with args.hash_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=HASH_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {
                    "instance": row["instance"],
                    "mps": row["mps"],
                    "sha256": sha256(args.suite_dir / row["mps"]),
                }
                for row in task_rows
            )
        print(
            f"input_hashes={len(task_rows)} sha256={sha256(args.hash_output)} "
            f"output={args.hash_output}"
        )


if __name__ == "__main__":
    main()
