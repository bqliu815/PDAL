#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from generate_synthetic_eqbox_qp_v2 import (
    SIZE_SPECS,
    sample_schedule,
    generate_a_matrix,
    generate_h_matrix,
    plant_primal_point,
    write_qps,
)

import numpy as np
import scipy.sparse as sp


A_FAMILIES = {
    "ill_scaled": {
        "alpha_r": 4.0,
        "alpha_c": 4.0,
        "p_r": 0.02,
        "p_c": 0.02,
        "gamma": 0.98,
        "description": "Severely scaled, weakly dependent",
    },
    "near_dependent_soft": {
        "alpha_r": 1.5,
        "alpha_c": 1.5,
        "p_r": 0.10,
        "p_c": 0.10,
        "gamma": 0.9990,
        "description": "Moderately dependent, milder than the original near_dependent family",
    },
    "hybrid_soft": {
        "alpha_r": 2.5,
        "alpha_c": 2.5,
        "p_r": 0.10,
        "p_c": 0.10,
        "gamma": 0.9990,
        "description": "Combined scaling and moderate dependence",
    },
}

H_ALIGN_FAMILIES = {
    "row_very_weak": {
        "row_log10_min": -8.0,
        "row_log10_max": -6.0,
        "null_log10_min": -1.0,
        "null_log10_max": 0.0,
        "jitter": 1e-10,
        "description": "Very weak curvature on range(A^T), strong curvature on null(A)",
    },
    "balanced": {
        "row_log10_min": -3.0,
        "row_log10_max": -1.0,
        "null_log10_min": -3.0,
        "null_log10_max": -1.0,
        "jitter": 1e-10,
        "description": "Balanced curvature on range(A^T) and null(A)",
    },
}

BOX_REGIMES = {
    "mostly_interior": {
        "lower_active_ratio": 0.01,
        "upper_active_ratio": 0.01,
        "multiplier_min": 0.01,
        "multiplier_max": 0.08,
        "description": "Equality-dominated with very few active bounds",
    },
    "moderately_active": {
        "lower_active_ratio": 0.08,
        "upper_active_ratio": 0.08,
        "multiplier_min": 0.05,
        "multiplier_max": 0.50,
        "description": "Moderate active-set regime with softer complementarity",
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

    return {
        "A": sp.csr_matrix(a_dense),
        "H": h,
        "b": b,
        "c": c,
        "x_star": x_star,
        "y_star": y_star,
        "s_star": s_star,
        "lb": np.full(spec.n, BOUND_LOWER, dtype=np.float64),
        "ub": np.full(spec.n, BOUND_UPPER, dtype=np.float64),
    }


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
        description="Generate a focused mechanism-driven synthetic equality-box QP benchmark."
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
        help="Number of instances for each family",
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
