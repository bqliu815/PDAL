# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Tests for the deterministic 856-task manifest builder."""

import csv
import subprocess
import sys
from pathlib import Path


def test_small_inventory_manifest_is_unique_and_ordered(tmp_path: Path) -> None:
    miplib_root = tmp_path / "miplib"
    for split in ("Large", "Medium", "Small"):
        directory = miplib_root / split
        directory.mkdir(parents=True)
        (directory / f"{split.lower()}.mps").touch()
    mittelmann_root = tmp_path / "mittelmann"
    mittelmann_root.mkdir()
    (mittelmann_root / "mittel.mps").touch()
    output = tmp_path / "manifest.tsv"

    script = Path(__file__).resolve().parents[1] / "make_full856_manifest.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--miplib-root",
            str(miplib_root),
            "--mittelmann-root",
            str(mittelmann_root),
            "--output",
            str(output),
            "--allow-count-mismatch",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    assert len(
        {
            (row["dataset"], row["split"], row["instance"], row["tolerance"])
            for row in rows
        }
    ) == 8
    assert [row["tolerance"] for row in rows[:4]] == ["1e-4"] * 4
    assert [row["split"] for row in rows[:4]] == [
        "Large",
        "Medium",
        "Small",
        "Mittelmann",
    ]
    assert "rows=8" in completed.stdout
