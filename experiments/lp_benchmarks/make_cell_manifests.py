#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Split the common 856-row LP manifest into the eight reporting cells."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 856:
        raise RuntimeError(f"expected 856 tasks, found {len(rows)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cell = row["split"] if row["dataset"] == "MIPLIB" else "Mittelmann"
        grouped[(cell, row["tolerance"])].append(row)

    expected = {
        ("Small", "1e-4"): 268,
        ("Medium", "1e-4"): 93,
        ("Large", "1e-4"): 18,
        ("Mittelmann", "1e-4"): 49,
        ("Small", "1e-8"): 268,
        ("Medium", "1e-8"): 93,
        ("Large", "1e-8"): 18,
        ("Mittelmann", "1e-8"): 49,
    }
    actual = {key: len(value) for key, value in grouped.items()}
    if actual != expected:
        raise RuntimeError(f"cell count mismatch: expected {expected}, found {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (cell, tolerance), cell_rows in grouped.items():
        output = args.output_dir / f"{cell}_eps{tolerance.replace('-', 'm')}.tsv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=cell_rows[0].keys(),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(cell_rows)
        print(f"{output}\t{len(cell_rows)}")


if __name__ == "__main__":
    main()
