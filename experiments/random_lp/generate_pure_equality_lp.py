#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Generate synthetic pure-equality box-LP instances and write them as MPS.

The generated LPs have the form

    min c^T x
    s.t. A x = b
         l <= x <= u

The construction used by the paper is fully specified in this file:

- sparse random A with safeguarded nonzero rows/cols
- injected near row/column dependence
- heterogeneous row/column scaling
- planted feasible point x_feas
- objective aligned with A^T y plus small noise

Outputs:

- `<instance>.mps` or `<instance>.mps.gz`
- `<instance>_meta.json`
- `manifest.csv` for the whole batch
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass
class InstanceMeta:
    instance: str
    seed: int
    m: int
    n: int
    nnz: int
    density_realized: float
    feasible_residual: float
    coefficient_dynamic_range: float
    row_scale_min: float
    row_scale_max: float
    col_scale_min: float
    col_scale_max: float
    x_feas_norm: float
    b_norm: float
    c_norm: float
    bound_active_fraction_realized: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pure-equality LP instances in MPS format."
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for MPS/meta files.")
    parser.add_argument("--name", default="peq_lp", help="Instance name or name prefix.")
    parser.add_argument("--count", type=int, default=1, help="Number of instances to generate.")
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed.")
    parser.add_argument("--m", type=int, default=200, help="Number of equality rows.")
    parser.add_argument("--n", type=int, default=300, help="Number of variables.")
    parser.add_argument("--density", type=float, default=0.05, help="Target matrix density.")
    parser.add_argument(
        "--row_scale_log10",
        type=float,
        default=5.0,
        help="Rows are rescaled by 10^u with u in [-row_scale_log10, row_scale_log10].",
    )
    parser.add_argument(
        "--col_scale_log10",
        type=float,
        default=5.0,
        help="Cols are rescaled by 10^u with u in [-col_scale_log10, col_scale_log10].",
    )
    parser.add_argument(
        "--bound_scale_log10",
        type=float,
        default=3.0,
        help="Variable box half-widths scale like 10^u with u in [-bound_scale_log10, bound_scale_log10].",
    )
    parser.add_argument(
        "--near_dep_col_fraction",
        type=float,
        default=0.18,
        help="Fraction of columns to overwrite with near-dependent copies.",
    )
    parser.add_argument(
        "--near_dep_row_fraction",
        type=float,
        default=0.18,
        help="Fraction of rows to overwrite with near-dependent copies.",
    )
    parser.add_argument(
        "--correlation_strength",
        type=float,
        default=0.995,
        help="Strength for injected near dependence.",
    )
    parser.add_argument(
        "--objective_noise",
        type=float,
        default=1e-6,
        help="Noise scale added to A^T y_seed in the objective.",
    )
    parser.add_argument(
        "--bound_active_fraction",
        type=float,
        default=0.30,
        help="Approximate fraction of variables placed exactly on bounds in x_feas.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Write compressed .mps.gz files instead of plain .mps.",
    )
    return parser.parse_args()


def random_sparse_matrix(m: int, n: int, density: float, rng: np.random.Generator) -> sparse.csr_matrix:
    nnz_target = max(1, int(round(m * n * density)))
    row_idx = rng.integers(0, m, size=nnz_target)
    col_idx = rng.integers(0, n, size=nnz_target)
    data = rng.standard_normal(nnz_target)
    matrix = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(m, n)).tocsr()
    matrix.sum_duplicates()
    return matrix


def ensure_nonzero_rows_and_cols(matrix: sparse.spmatrix, rng: np.random.Generator) -> sparse.csr_matrix:
    lil = matrix.tolil(copy=True)
    m, n = lil.shape
    row_nnz = np.asarray(lil.getnnz(axis=1)).ravel()
    col_nnz = np.asarray(lil.getnnz(axis=0)).ravel()

    for i in np.where(row_nnz == 0)[0]:
        j = int(rng.integers(0, n))
        lil[i, j] = float(rng.standard_normal())
    for j in np.where(col_nnz == 0)[0]:
        i = int(rng.integers(0, m))
        lil[i, j] = float(rng.standard_normal())
    return lil.tocsr()


def sparse_random_vector(length: int, density: float, rng: np.random.Generator) -> sparse.csr_matrix:
    density = float(min(1.0, max(0.0, density)))
    nnz_target = max(1, int(round(length * density)))
    idx = rng.integers(0, length, size=nnz_target)
    data = rng.standard_normal(nnz_target)
    vec = sparse.coo_matrix((data, (idx, np.zeros(nnz_target, dtype=int))), shape=(length, 1)).tocsr()
    vec.sum_duplicates()
    return vec


def inject_column_dependence(matrix: sparse.csr_matrix, fraction: float, strength: float, rng: np.random.Generator) -> sparse.csr_matrix:
    m, n = matrix.shape
    num_pairs = max(1, int(round(fraction * n)))
    lil = matrix.tolil(copy=True)
    for _ in range(num_pairs):
        src = int(rng.integers(0, n))
        dst = int(rng.integers(0, n))
        if src == dst:
            continue
        src_nnz = lil[:, src].nnz
        density = min(0.2, max(0.02, src_nnz / max(m, 1)))
        perturb = sparse_random_vector(m, density, rng).tolil()
        lil[:, dst] = strength * lil[:, src] + (1.0 - strength) * perturb
    return lil.tocsr()


def inject_row_dependence(matrix: sparse.csr_matrix, fraction: float, strength: float, rng: np.random.Generator) -> sparse.csr_matrix:
    m, n = matrix.shape
    num_pairs = max(1, int(round(fraction * m)))
    lil = matrix.tolil(copy=True)
    for _ in range(num_pairs):
        src = int(rng.integers(0, m))
        dst = int(rng.integers(0, m))
        if src == dst:
            continue
        src_nnz = lil.getrow(src).nnz
        density = min(0.2, max(0.02, src_nnz / max(n, 1)))
        perturb = sparse_random_vector(n, density, rng).T.tolil()
        lil[dst, :] = strength * lil.getrow(src) + (1.0 - strength) * perturb
    return lil.tocsr()


def plant_box_point(n: int, bound_scale_log10: float, bound_active_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_span = np.power(10.0, (2.0 * rng.random(n) - 1.0) * bound_scale_log10)
    var_lb = -box_span
    var_ub = box_span
    x_feas = var_lb + rng.random(n) * (var_ub - var_lb)

    active_lower = rng.random(n) < 0.5 * bound_active_fraction
    active_upper = (~active_lower) & (rng.random(n) < 0.5 * bound_active_fraction)
    x_feas[active_lower] = var_lb[active_lower]
    x_feas[active_upper] = var_ub[active_upper]
    return var_lb, var_ub, x_feas


def max_box_violation(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    return float(np.max(np.maximum.reduce([lb - x, x - ub, np.zeros_like(x)])))


def coefficient_dynamic_range(matrix: sparse.csr_matrix) -> float:
    abs_vals = np.abs(matrix.data)
    if abs_vals.size == 0:
        return 1.0
    return float(abs_vals.max() / max(abs_vals.min(), np.finfo(float).eps))


def generate_instance(args: argparse.Namespace, seed: int, name: str) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, InstanceMeta]:
    rng = np.random.default_rng(seed)
    A = random_sparse_matrix(args.m, args.n, args.density, rng)
    A = ensure_nonzero_rows_and_cols(A, rng)
    A = inject_column_dependence(A, args.near_dep_col_fraction, args.correlation_strength, rng)
    A = inject_row_dependence(A, args.near_dep_row_fraction, args.correlation_strength, rng)

    row_scale = np.power(10.0, (2.0 * rng.random(args.m) - 1.0) * args.row_scale_log10)
    col_scale = np.power(10.0, (2.0 * rng.random(args.n) - 1.0) * args.col_scale_log10)
    A = sparse.diags(row_scale) @ A @ sparse.diags(col_scale)
    A = ensure_nonzero_rows_and_cols(A, rng)

    var_lb, var_ub, x_feas = plant_box_point(args.n, args.bound_scale_log10, args.bound_active_fraction, rng)
    b = np.asarray(A @ x_feas).ravel()
    y_seed = rng.standard_normal(args.m) * np.power(10.0, 2.0 * rng.random(args.m) - 1.0)
    c = np.asarray(A.T @ y_seed).ravel() + args.objective_noise * col_scale * rng.standard_normal(args.n)

    residual = max(float(np.max(np.abs(A @ x_feas - b))), max_box_violation(x_feas, var_lb, var_ub))
    active_realized = float(np.mean((np.isclose(x_feas, var_lb)) | (np.isclose(x_feas, var_ub))))
    meta = InstanceMeta(
        instance=name,
        seed=seed,
        m=args.m,
        n=args.n,
        nnz=int(A.nnz),
        density_realized=float(A.nnz / max(args.m * args.n, 1)),
        feasible_residual=residual,
        coefficient_dynamic_range=coefficient_dynamic_range(A),
        row_scale_min=float(row_scale.min()),
        row_scale_max=float(row_scale.max()),
        col_scale_min=float(col_scale.min()),
        col_scale_max=float(col_scale.max()),
        x_feas_norm=float(np.linalg.norm(x_feas)),
        b_norm=float(np.linalg.norm(b)),
        c_norm=float(np.linalg.norm(c)),
        bound_active_fraction_realized=active_realized,
    )
    return A.tocsr(), b, c, var_lb, var_ub, meta


def row_name(i: int) -> str:
    return f"R{i + 1:07d}"


def col_name(j: int) -> str:
    return f"X{j + 1:07d}"


def format_float(value: float) -> str:
    return f"{value:.16g}"


def paired_entries(entries: list[tuple[str, float]]) -> Iterable[tuple[tuple[str, float], tuple[str, float] | None]]:
    idx = 0
    while idx < len(entries):
        first = entries[idx]
        second = entries[idx + 1] if idx + 1 < len(entries) else None
        yield first, second
        idx += 2


def write_mps(path: Path, name: str, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, var_lb: np.ndarray, var_ub: np.ndarray) -> None:
    csc = A.tocsc()
    lines: list[str] = []
    lines.append(f"NAME          {name[:8].upper()}")
    lines.append("ROWS")
    lines.append(" N  OBJ")
    for i in range(A.shape[0]):
        lines.append(f" E  {row_name(i)}")

    lines.append("COLUMNS")
    for j in range(A.shape[1]):
        entries: list[tuple[str, float]] = []
        if c[j] != 0.0:
            entries.append(("OBJ", float(c[j])))
        start, end = csc.indptr[j], csc.indptr[j + 1]
        rows = csc.indices[start:end]
        vals = csc.data[start:end]
        for i, val in zip(rows, vals, strict=True):
            if val != 0.0:
                entries.append((row_name(int(i)), float(val)))
        if not entries:
            continue
        var = col_name(j)
        for first, second in paired_entries(entries):
            if second is None:
                lines.append(
                    f"    {var:<8}  {first[0]:<8}  {format_float(first[1])}"
                )
            else:
                lines.append(
                    f"    {var:<8}  {first[0]:<8}  {format_float(first[1])}  {second[0]:<8}  {format_float(second[1])}"
                )

    lines.append("RHS")
    rhs_entries = [(row_name(i), float(bi)) for i, bi in enumerate(b) if bi != 0.0]
    for first, second in paired_entries(rhs_entries):
        if second is None:
            lines.append(f"    RHS1      {first[0]:<8}  {format_float(first[1])}")
        else:
            lines.append(
                f"    RHS1      {first[0]:<8}  {format_float(first[1])}  {second[0]:<8}  {format_float(second[1])}"
            )

    lines.append("BOUNDS")
    for j in range(A.shape[1]):
        var = col_name(j)
        lb = float(var_lb[j])
        ub = float(var_ub[j])
        if np.isfinite(lb) and np.isfinite(ub) and np.isclose(lb, ub):
            lines.append(f" FX BND1      {var:<8}  {format_float(lb)}")
            continue
        if not np.isfinite(lb) and not np.isfinite(ub):
            lines.append(f" FR BND1      {var:<8}")
            continue
        if np.isfinite(lb):
            if lb != 0.0:
                lines.append(f" LO BND1      {var:<8}  {format_float(lb)}")
        else:
            lines.append(f" MI BND1      {var:<8}")
        if np.isfinite(ub):
            lines.append(f" UP BND1      {var:<8}  {format_float(ub)}")

    lines.append("ENDATA")
    text = "\n".join(lines) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    width = max(3, len(str(args.count)))

    for idx in range(args.count):
        seed = args.seed + idx
        if args.count == 1:
            instance_name = args.name
        else:
            instance_name = f"{args.name}_{idx + 1:0{width}d}"

        A, b, c, var_lb, var_ub, meta = generate_instance(args, seed, instance_name)
        suffix = ".mps.gz" if args.gzip else ".mps"
        mps_path = out_dir / f"{instance_name}{suffix}"
        meta_path = out_dir / f"{instance_name}_meta.json"

        write_mps(mps_path, instance_name.upper(), A, b, c, var_lb, var_ub)
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

        row = asdict(meta)
        row["mps_path"] = str(mps_path)
        row["meta_path"] = str(meta_path)
        manifest_rows.append(row)

        print(
            f"[{idx + 1}/{args.count}] {instance_name}: "
            f"m={meta.m} n={meta.n} nnz={meta.nnz} "
            f"density={meta.density_realized:.3e} residual={meta.feasible_residual:.3e}"
        )

    manifest_path = out_dir / "manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
