#!/usr/bin/env python3
"""Build the deterministic MIPLIB/Mittelmann full856 benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


TOLERANCES = ("1e-4", "1e-8")
MIPLIB_SPLITS = ("Large", "Medium", "Small")
EXPECTED = {"Large": 18, "Medium": 93, "Small": 268, "Mittelmann": 49}
FIELDS = ("dataset", "split", "instance", "tolerance", "time_limit", "mps")


def mps_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise RuntimeError(f"benchmark directory does not exist: {directory}")
    files = sorted(
        (path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mps"),
        key=lambda path: path.stem,
    )
    names = [path.stem for path in files]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate MPS stems in {directory}")
    return files


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miplib-root", type=Path, required=True)
    parser.add_argument("--mittelmann-root", type=Path, required=True)
    parser.add_argument("--mittelmann-label", default="Mittelmann")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    miplib = {split: mps_files(args.miplib_root / split) for split in MIPLIB_SPLITS}
    mittelmann = mps_files(args.mittelmann_root)
    actual = {split: len(files) for split, files in miplib.items()}
    actual["Mittelmann"] = len(mittelmann)
    if not args.allow_count_mismatch and actual != EXPECTED:
        raise RuntimeError(f"benchmark count mismatch: expected {EXPECTED}, found {actual}")

    rows: list[dict[str, str | int]] = []
    for tolerance in TOLERANCES:
        for split in MIPLIB_SPLITS:
            limit = 18000 if split == "Large" else 3600
            rows.extend(
                {
                    "dataset": "MIPLIB",
                    "split": split,
                    "instance": path.stem,
                    "tolerance": tolerance,
                    "time_limit": limit,
                    "mps": str(path),
                }
                for path in miplib[split]
            )
        rows.extend(
            {
                "dataset": "MITTELMANN",
                "split": args.mittelmann_label,
                "instance": path.stem,
                "tolerance": tolerance,
                "time_limit": 1000,
                "mps": str(path),
            }
            for path in mittelmann
        )

    identities = {
        (row["dataset"], row["split"], row["instance"], row["tolerance"])
        for row in rows
    }
    expected_rows = 2 * sum(actual.values())
    if len(rows) != expected_rows or len(identities) != expected_rows:
        raise RuntimeError(
            "manifest rows must be unique: "
            f"expected {expected_rows}, found {len(rows)}/{len(identities)}"
        )
    if not args.allow_count_mismatch and expected_rows != 856:
        raise RuntimeError(f"benchmark manifest must contain 856 rows, found {expected_rows}")
    if any(not Path(str(row["mps"])).is_file() for row in rows):
        raise RuntimeError("manifest contains an unreadable MPS path")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={sha256(args.output)} output={args.output}")


if __name__ == "__main__":
    main()
