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

from generate_synthetic_eqbox_qp_v2 import generate_a_matrix, write_qps
from generate_synthetic_eqbox_qp_v2b import BOUND_LOWER, BOUND_UPPER, BOX_REGIMES, plant_primal_point


A_FAMILIES = {
    "well": {
        "alpha_r": 1.2,
        "alpha_c": 1.2,
        "p_r": 0.01,
        "p_c": 0.01,
        "gamma": 0.97,
        "description": "Well-scaled weakly dependent constraints",
    },
    "ill_scaled": {
        "alpha_r": 4.0,
        "alpha_c": 4.0,
        "p_r": 0.02,
        "p_c": 0.02,
        "gamma": 0.98,
        "description": "Severely scaled weakly dependent constraints",
    },
    "near_dependent_soft": {
        "alpha_r": 1.5,
        "alpha_c": 1.5,
        "p_r": 0.10,
        "p_c": 0.10,
        "gamma": 0.9990,
        "description": "Moderately dependent constraints",
    },
}


SIZE_SPECS = {
    "medium": {"m": 300, "n": 1200, "a_density": 0.010, "h_bandwidth": 5},
    "large": {"m": 520, "n": 2600, "a_density": 0.0045, "h_bandwidth": 5},
    "xlarge": {"m": 650, "n": 3600, "a_density": 0.0032, "h_bandwidth": 5},
}

SIZE_SPLIT = {"medium": 0.40, "large": 0.40, "xlarge": 0.20}

H_FAMILIES = {
    "banded_moderate": {
        "kind": "banded",
        "diag_low": 0.4,
        "diag_high": 1.4,
        "offdiag_scale": 0.03,
        "description": "Banded sparse SPD Hessian with moderate curvature",
    },
    "sparse_moderate": {
        "kind": "sparse_dd",
        "diag_low": 0.5,
        "diag_high": 1.8,
        "offdiag_scale": 0.05,
        "extra_density": 0.0015,
        "description": "Sparse diagonally dominant SPD Hessian with moderate curvature",
    },
}

REGIME_FAMILIES = {
    "linearized_large": [
        ("banded_moderate", "well", "mostly_interior"),
        ("banded_moderate", "well", "moderately_active"),
        ("sparse_moderate", "well", "mostly_interior"),
        ("sparse_moderate", "well", "moderately_active"),
    ],
}


@dataclass(frozen=True)
class InstanceSpec:
    regime: str
    family_key: str
    h_family: str
    a_family: str
    box_regime: str
    size_class: str
    local_index: int
    global_index: int
    m: int
    n: int
    a_density: float
    h_bandwidth: int


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
    for size_class in ("medium", "large", "xlarge"):
        schedule.extend([size_class] * counts[size_class])
    return schedule


def generate_h_matrix(rng: np.random.Generator, n: int, bandwidth: int, family: dict[str, float | str]) -> sp.csr_matrix:
    diag = rng.uniform(float(family["diag_low"]), float(family["diag_high"]), size=n)
    if family["kind"] == "banded":
        h = sp.diags(diag, offsets=0, format="lil")
        for k in range(1, bandwidth + 1):
            vals = rng.uniform(-float(family["offdiag_scale"]), float(family["offdiag_scale"]), size=n - k)
            h.setdiag(vals, k=k)
            h.setdiag(vals, k=-k)
        off_abs = np.asarray(np.abs(h).sum(axis=1)).ravel() - np.abs(h.diagonal())
        h.setdiag(diag + off_abs + 0.05)
        return h.tocsr()

    target_edges = max(n - 1, int(round(float(family["extra_density"]) * n * max(n - 1, 1) / 2.0)))
    rows = list(range(n - 1))
    cols = list(range(1, n))
    vals = list(rng.uniform(-float(family["offdiag_scale"]), float(family["offdiag_scale"]), size=n - 1))
    seen = {(i, i + 1) for i in range(n - 1)}
    for _ in range(max(0, target_edges - (n - 1))):
        for _attempt in range(64):
            i = int(rng.integers(0, n - 1))
            j = int(rng.integers(i + 1, min(n, i + 1 + bandwidth)))
            if (i, j) not in seen:
                seen.add((i, j))
                rows.append(i)
                cols.append(j)
                vals.append(float(rng.uniform(-float(family["offdiag_scale"]), float(family["offdiag_scale"]))))
                break
    off = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    off = off + off.T
    off_abs = np.asarray(np.abs(off).sum(axis=1)).ravel()
    return (sp.diags(diag + off_abs + 0.05, format="csr") + off.tocsr()).tocsr()


def build_instance(spec: InstanceSpec, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    a = generate_a_matrix(rng, spec.m, spec.n, spec.a_density, A_FAMILIES[spec.a_family])
    h = generate_h_matrix(rng, spec.n, spec.h_bandwidth, H_FAMILIES[spec.h_family])
    x_star, s_star = plant_primal_point(rng, spec.n, BOX_REGIMES[spec.box_regime])
    y_star = rng.normal(scale=0.30, size=spec.m)
    b = a @ x_star
    c = a.T @ y_star - h @ x_star - s_star
    return {
        "A": sp.csr_matrix(a),
        "H": h,
        "b": b,
        "c": c,
        "lb": np.full(spec.n, BOUND_LOWER),
        "ub": np.full(spec.n, BOUND_UPPER),
    }


def generate_dataset(out_dir: Path, regime: str, total_instances: int, seed: int, force: bool) -> None:
    if regime not in REGIME_FAMILIES:
        raise SystemExit(f"Unknown regime {regime!r}; choose from {sorted(REGIME_FAMILIES)}")
    families = REGIME_FAMILIES[regime]
    if total_instances % len(families) != 0:
        raise SystemExit(f"--total-instances must be divisible by {len(families)} for regime {regime}")
    per_family = total_instances // len(families)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(f"Output dir already exists and is non-empty: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "instances").mkdir(parents=True)
    (out_dir / "lists").mkdir()

    rows: list[dict[str, object]] = []
    all_paths: list[str] = []
    global_index = 0
    schedule = sample_schedule(per_family)
    for h_family, a_family, box_regime in families:
        family_key = f"{regime}__{h_family}__{a_family}__{box_regime}"
        family_dir = out_dir / "instances" / family_key
        family_dir.mkdir()
        family_paths: list[str] = []
        for local_index, size_class in enumerate(schedule, start=1):
            size = SIZE_SPECS[size_class]
            spec = InstanceSpec(
                regime=regime,
                family_key=family_key,
                h_family=h_family,
                a_family=a_family,
                box_regime=box_regime,
                size_class=size_class,
                local_index=local_index,
                global_index=global_index,
                m=int(size["m"]),
                n=int(size["n"]),
                a_density=float(size["a_density"]),
                h_bandwidth=int(size["h_bandwidth"]),
            )
            instance_seed = seed + 104729 * global_index + 1009
            data = build_instance(spec, instance_seed)
            name = f"{family_key}__{size_class}__{local_index:04d}"
            qps_path = family_dir / f"{name}.QPS"
            write_qps(
                qps_path,
                name=name[:60],
                a=data["A"],
                h=data["H"],
                b=data["b"],
                c=data["c"],
                lb=data["lb"],
                ub=data["ub"],
            )
            abs_path = str(qps_path.resolve())
            family_paths.append(abs_path)
            all_paths.append(abs_path)
            rows.append(
                {
                    "regime": regime,
                    "family_key": family_key,
                    "h_family": h_family,
                    "a_family": a_family,
                    "box_regime": box_regime,
                    "size_class": size_class,
                    "local_index": local_index,
                    "global_index": global_index,
                    "seed": instance_seed,
                    "m": spec.m,
                    "n": spec.n,
                    "a_density": spec.a_density,
                    "instance_name": name,
                    "qps_path": abs_path,
                }
            )
            global_index += 1
        (out_dir / "lists" / f"{family_key}.txt").write_text("\n".join(family_paths) + "\n")
    (out_dir / "lists" / "all_instances.txt").write_text("\n".join(all_paths) + "\n")

    fieldnames = list(rows[0])
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": seed,
        "regime": regime,
        "total_instances": total_instances,
        "instances_per_family": per_family,
        "families": families,
        "size_specs": SIZE_SPECS,
        "size_split": SIZE_SPLIT,
        "a_families": A_FAMILIES,
        "h_families": H_FAMILIES,
        "box_regimes": BOX_REGIMES,
        "construction": "planted KKT: b=A x_star and c=A^T y_star - H x_star - s_star",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified planted-KKT equality-box QP datasets.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--regime", default="linearized_large", choices=sorted(REGIME_FAMILIES))
    parser.add_argument("--total-instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_dataset(args.out_dir.expanduser().resolve(), args.regime, args.total_instances, args.seed, args.force)
    print(f"Wrote dataset to {args.out_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
