#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SIZE_SPECS = {
    "small": {"m": 60, "n": 160, "a_density": 0.060},
    "medium": {"m": 100, "n": 240, "a_density": 0.040},
    "large": {"m": 160, "n": 320, "a_density": 0.025},
}

SIZE_SPLIT = {
    "small": 0.4,
    "medium": 0.4,
    "large": 0.2,
}

A_FAMILIES = {
    "well": {
        "alpha_r": 1.5,
        "alpha_c": 1.5,
        "p_r": 0.02,
        "p_c": 0.02,
        "gamma": 0.98,
        "description": "Well-scaled, weakly dependent",
    },
    "ill_scaled": {
        "alpha_r": 4.5,
        "alpha_c": 4.5,
        "p_r": 0.02,
        "p_c": 0.02,
        "gamma": 0.98,
        "description": "Severely scaled, weakly dependent",
    },
    "near_dependent": {
        "alpha_r": 1.5,
        "alpha_c": 1.5,
        "p_r": 0.26,
        "p_c": 0.26,
        "gamma": 0.9995,
        "description": "Well-scaled, strongly dependent",
    },
}

H_ALIGN_FAMILIES = {
    "row_weak": {
        "row_log10_min": -6.0,
        "row_log10_max": -4.0,
        "null_log10_min": -1.5,
        "null_log10_max": -0.2,
        "jitter": 1e-8,
        "description": "Weak curvature on range(A^T), stronger curvature on null(A)",
    },
    "balanced": {
        "row_log10_min": -2.5,
        "row_log10_max": -0.8,
        "null_log10_min": -2.5,
        "null_log10_max": -0.8,
        "jitter": 1e-8,
        "description": "Comparable curvature on range(A^T) and null(A)",
    },
    "null_weak": {
        "row_log10_min": -1.5,
        "row_log10_max": -0.2,
        "null_log10_min": -6.0,
        "null_log10_max": -4.0,
        "jitter": 1e-8,
        "description": "Stronger curvature on range(A^T), weak curvature on null(A)",
    },
}

BOX_REGIMES = {
    "mostly_interior": {
        "lower_active_ratio": 0.02,
        "upper_active_ratio": 0.02,
        "multiplier_min": 0.02,
        "multiplier_max": 0.20,
        "description": "Equality-dominated regime with only a few active bounds",
    },
    "moderately_active": {
        "lower_active_ratio": 0.12,
        "upper_active_ratio": 0.12,
        "multiplier_min": 0.10,
        "multiplier_max": 1.00,
        "description": "Mixed regime with a moderate number of active bounds",
    },
}

BOUND_LOWER = -1.0
BOUND_UPPER = 1.0


@dataclass(frozen=True)
class InstanceSpec:
    family_key: str
    h_align_family: str
    a_family: str
    box_regime: str
    size_class: str
    local_index: int
    global_index: int
    m: int
    n: int
    a_density: float


def allocate_size_counts(total: int) -> dict[str, int]:
    counts = {}
    assigned = 0
    residuals: list[tuple[float, str]] = []
    for size_class, frac in SIZE_SPLIT.items():
        raw = total * frac
        base = int(math.floor(raw))
        counts[size_class] = base
        assigned += base
        residuals.append((raw - base, size_class))
    for _, size_class in sorted(residuals, reverse=True):
        if assigned >= total:
            break
        counts[size_class] += 1
        assigned += 1
    return counts


def sample_schedule(instances_per_family: int) -> list[str]:
    counts = allocate_size_counts(instances_per_family)
    schedule: list[str] = []
    for size_class in ("small", "medium", "large"):
        schedule.extend([size_class] * counts[size_class])
    return schedule


def _nonzero_noise(rng: np.random.Generator, length: int, nnz: int) -> np.ndarray:
    vec = np.zeros(length, dtype=np.float64)
    nnz = max(1, min(length, nnz))
    idx = rng.choice(length, size=nnz, replace=False)
    vec[idx] = rng.normal(size=nnz)
    norm = np.linalg.norm(vec)
    if not math.isfinite(norm) or norm == 0.0:
        vec[idx[0]] = 1.0
        norm = 1.0
    return vec / norm


def generate_a_matrix(
    rng: np.random.Generator,
    m: int,
    n: int,
    density: float,
    family: dict[str, float],
) -> np.ndarray:
    a = np.zeros((m, n), dtype=np.float64)
    mask = rng.random((m, n)) < density
    values = rng.uniform(-1.0, 1.0, size=int(mask.sum()))
    a[mask] = values

    for row in np.where(np.all(a == 0.0, axis=1))[0]:
        col = int(rng.integers(0, n))
        a[row, col] = rng.uniform(-1.0, 1.0)
    for col in np.where(np.all(a == 0.0, axis=0))[0]:
        row = int(rng.integers(0, m))
        a[row, col] = rng.uniform(-1.0, 1.0)

    gamma = float(family["gamma"])
    dep_scale = math.sqrt(max(1.0 - gamma * gamma, 1e-10))

    num_rows = int(round(family["p_r"] * m))
    if num_rows > 0:
        rows = rng.choice(m, size=num_rows, replace=False)
        row_nnz = max(1, int(round(density * n)))
        for row in rows:
            src = int(rng.integers(0, m - 1))
            if src >= row:
                src += 1
            base = a[src, :]
            noise = _nonzero_noise(rng, n, row_nnz)
            base_norm = np.linalg.norm(base)
            if base_norm == 0.0:
                base = _nonzero_noise(rng, n, row_nnz)
                base_norm = 1.0
            a[row, :] = gamma * base + dep_scale * base_norm * noise

    num_cols = int(round(family["p_c"] * n))
    if num_cols > 0:
        cols = rng.choice(n, size=num_cols, replace=False)
        col_nnz = max(1, int(round(density * m)))
        for col in cols:
            src = int(rng.integers(0, n - 1))
            if src >= col:
                src += 1
            base = a[:, src]
            noise = _nonzero_noise(rng, m, col_nnz)
            base_norm = np.linalg.norm(base)
            if base_norm == 0.0:
                base = _nonzero_noise(rng, m, col_nnz)
                base_norm = 1.0
            a[:, col] = gamma * base + dep_scale * base_norm * noise

    row_scales = np.power(10.0, rng.uniform(-family["alpha_r"], family["alpha_r"], size=m))
    col_scales = np.power(10.0, rng.uniform(-family["alpha_c"], family["alpha_c"], size=n))
    return row_scales[:, None] * a * col_scales[None, :]


def orthogonal_decomposition(a_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, singular_vals, vt = np.linalg.svd(a_dense, full_matrices=True)
    if singular_vals.size == 0:
        q_row = np.zeros((a_dense.shape[1], 0), dtype=np.float64)
        q_null = np.eye(a_dense.shape[1], dtype=np.float64)
        return q_row, q_null, singular_vals
    tol = np.max(singular_vals) * max(a_dense.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular_vals > tol))
    v = vt.T
    q_row = v[:, :rank]
    q_null = v[:, rank:]
    return q_row, q_null, singular_vals


def generate_h_matrix(
    rng: np.random.Generator,
    a_dense: np.ndarray,
    family: dict[str, float],
) -> sp.csr_matrix:
    q_row, q_null, _ = orthogonal_decomposition(a_dense)
    n = a_dense.shape[1]

    h = np.zeros((n, n), dtype=np.float64)
    if q_row.shape[1] > 0:
        row_eigs = np.power(
            10.0,
            rng.uniform(family["row_log10_min"], family["row_log10_max"], size=q_row.shape[1]),
        )
        h += (q_row * row_eigs) @ q_row.T
    if q_null.shape[1] > 0:
        null_eigs = np.power(
            10.0,
            rng.uniform(family["null_log10_min"], family["null_log10_max"], size=q_null.shape[1]),
        )
        h += (q_null * null_eigs) @ q_null.T

    diag_jitter = np.power(10.0, rng.uniform(-6.0, -4.0, size=n))
    h += np.diag(diag_jitter + family["jitter"])
    h = 0.5 * (h + h.T)
    return sp.csr_matrix(h)


def plant_primal_point(
    rng: np.random.Generator,
    n: int,
    regime: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-0.6, 0.6, size=n)
    num_lower = int(round(regime["lower_active_ratio"] * n))
    num_upper = int(round(regime["upper_active_ratio"] * n))
    perm = rng.permutation(n)
    lower_idx = perm[:num_lower]
    upper_idx = perm[num_lower : num_lower + num_upper]
    x[lower_idx] = BOUND_LOWER
    x[upper_idx] = BOUND_UPPER

    s = np.zeros(n, dtype=np.float64)
    if num_lower > 0:
        s[lower_idx] = -rng.uniform(regime["multiplier_min"], regime["multiplier_max"], size=num_lower)
    if num_upper > 0:
        s[upper_idx] = rng.uniform(regime["multiplier_min"], regime["multiplier_max"], size=num_upper)
    return x, s


def build_instance(
    spec: InstanceSpec,
    family_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(family_seed)
    a_family = A_FAMILIES[spec.a_family]
    h_family = H_ALIGN_FAMILIES[spec.h_align_family]
    box_regime = BOX_REGIMES[spec.box_regime]

    a_dense = generate_a_matrix(rng, spec.m, spec.n, spec.a_density, a_family)
    h = generate_h_matrix(rng, a_dense, h_family)
    x_star, s_star = plant_primal_point(rng, spec.n, box_regime)
    y_star = rng.normal(scale=0.35, size=spec.m)
    b = a_dense @ x_star
    h_x = h @ x_star
    c = a_dense.T @ y_star - h_x - s_star

    a_sparse = sp.csr_matrix(a_dense)
    return {
        "A": a_sparse,
        "H": h,
        "b": b,
        "c": c,
        "x_star": x_star,
        "y_star": y_star,
        "s_star": s_star,
        "lb": np.full(spec.n, BOUND_LOWER, dtype=np.float64),
        "ub": np.full(spec.n, BOUND_UPPER, dtype=np.float64),
    }


def fmt_num(value: float) -> str:
    if not math.isfinite(value):
        if value > 0:
            return "INF"
        if value < 0:
            return "-INF"
        return "0"
    return f"{value:.16g}"


def write_qps(
    path: Path,
    name: str,
    a: sp.csr_matrix,
    h: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> None:
    m, n = a.shape
    row_names = ["OBJ"] + [f"C{i + 1:05d}" for i in range(m)]
    col_names = [f"X{i + 1:05d}" for i in range(n)]

    with path.open("w", encoding="ascii") as fh:
        fh.write(f"NAME {name}\n")
        fh.write("OBJSENSE\n")
        fh.write(" MIN\n")
        fh.write("ROWS\n")
        fh.write(" N OBJ\n")
        for row_name in row_names[1:]:
            fh.write(f" E {row_name}\n")

        fh.write("COLUMNS\n")
        a_csc = a.tocsc()
        for j, col_name in enumerate(col_names):
            if c[j] != 0.0:
                fh.write(f" {col_name} OBJ {fmt_num(float(c[j]))}\n")
            start = int(a_csc.indptr[j])
            end = int(a_csc.indptr[j + 1])
            for idx in range(start, end):
                row = int(a_csc.indices[idx])
                value = float(a_csc.data[idx])
                if value == 0.0:
                    continue
                fh.write(f" {col_name} {row_names[row + 1]} {fmt_num(value)}\n")

        fh.write("RHS\n")
        for i, rhs in enumerate(b):
            fh.write(f" RHS1 {row_names[i + 1]} {fmt_num(float(rhs))}\n")

        fh.write("BOUNDS\n")
        for j, col_name in enumerate(col_names):
            fh.write(f" LO BND1 {col_name} {fmt_num(float(lb[j]))}\n")
            fh.write(f" UP BND1 {col_name} {fmt_num(float(ub[j]))}\n")

        fh.write("QUADOBJ\n")
        h_coo = sp.triu(h, format="coo")
        for i, j, value in zip(h_coo.row, h_coo.col, h_coo.data):
            val = float(value)
            if val == 0.0:
                continue
            fh.write(f" {col_names[int(i)]} {col_names[int(j)]} {fmt_num(val)}\n")

        fh.write("ENDATA\n")


def generate_dataset(out_dir: Path, instances_per_family: int, seed: int, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(f"Output dir already exists and is non-empty: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_dir = out_dir / "instances"
    lists_dir = out_dir / "lists"
    instances_dir.mkdir()
    lists_dir.mkdir()

    schedule = sample_schedule(instances_per_family)
    manifest_rows: list[dict[str, object]] = []
    all_instance_paths: list[str] = []
    family_lists: dict[str, list[str]] = {}
    global_index = 0

    for h_align_family in H_ALIGN_FAMILIES:
        for a_family in A_FAMILIES:
            for box_regime in BOX_REGIMES:
                family_key = f"{h_align_family}__{a_family}__{box_regime}"
                family_dir = instances_dir / family_key
                family_dir.mkdir()
                family_lists[family_key] = []
                for local_index, size_class in enumerate(schedule, start=1):
                    size_spec = SIZE_SPECS[size_class]
                    spec = InstanceSpec(
                        family_key=family_key,
                        h_align_family=h_align_family,
                        a_family=a_family,
                        box_regime=box_regime,
                        size_class=size_class,
                        local_index=local_index,
                        global_index=global_index,
                        m=size_spec["m"],
                        n=size_spec["n"],
                        a_density=size_spec["a_density"],
                    )
                    instance_seed = seed + 104729 * global_index + 1009
                    data = build_instance(spec, instance_seed)
                    instance_name = f"{family_key}__{size_class}__{local_index:03d}"
                    qps_path = family_dir / f"{instance_name}.QPS"
                    write_qps(
                        qps_path,
                        name=instance_name[:60],
                        a=data["A"],
                        h=data["H"],
                        b=data["b"],
                        c=data["c"],
                        lb=data["lb"],
                        ub=data["ub"],
                    )
                    abs_path = str(qps_path.resolve())
                    family_lists[family_key].append(abs_path)
                    all_instance_paths.append(abs_path)
                    manifest_rows.append(
                        {
                            "family_key": family_key,
                            "h_align_family": h_align_family,
                            "a_family": a_family,
                            "box_regime": box_regime,
                            "size_class": size_class,
                            "local_index": local_index,
                            "global_index": global_index,
                            "seed": instance_seed,
                            "m": spec.m,
                            "n": spec.n,
                            "a_density": spec.a_density,
                            "instance_name": instance_name,
                            "qps_path": abs_path,
                        }
                    )
                    global_index += 1

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "family_key",
                "h_align_family",
                "a_family",
                "box_regime",
                "size_class",
                "local_index",
                "global_index",
                "seed",
                "m",
                "n",
                "a_density",
                "instance_name",
                "qps_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    for family_key, paths in family_lists.items():
        with (lists_dir / f"{family_key}.txt").open("w", encoding="utf-8") as fh:
            for path in paths:
                fh.write(path + "\n")

    with (lists_dir / "all_instances.txt").open("w", encoding="utf-8") as fh:
        for path in all_instance_paths:
            fh.write(path + "\n")

    metadata = {
        "seed": seed,
        "instances_per_family": instances_per_family,
        "num_h_align_families": len(H_ALIGN_FAMILIES),
        "num_a_families": len(A_FAMILIES),
        "num_box_regimes": len(BOX_REGIMES),
        "num_cross_families": len(H_ALIGN_FAMILIES) * len(A_FAMILIES) * len(BOX_REGIMES),
        "num_total_instances": len(manifest_rows),
        "size_specs": SIZE_SPECS,
        "size_split": SIZE_SPLIT,
        "h_align_families": H_ALIGN_FAMILIES,
        "a_families": A_FAMILIES,
        "box_regimes": BOX_REGIMES,
        "box_bounds": {"lower": BOUND_LOWER, "upper": BOUND_UPPER},
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mechanism-driven synthetic equality-box QP instances with H-A alignment families."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output dataset root",
    )
    parser.add_argument(
        "--instances-per-family",
        type=int,
        default=20,
        help="Number of instances for each H_align x A x box_regime family",
    )
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    generate_dataset(
        out_dir=out_dir,
        instances_per_family=args.instances_per_family,
        seed=args.seed,
        force=args.force,
    )
    print(f"Wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
