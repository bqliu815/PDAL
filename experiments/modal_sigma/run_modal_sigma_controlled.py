#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Controlled validation of the FA-CP modal penalty prediction.

The experiment constructs equality-constrained QPs for which H and A^T A are
exactly simultaneously diagonalizable.  A penalty is predicted from the modal
spectral envelope before any trajectory is generated.  Exact FA-CP trajectories
in the original coordinates are then used to estimate the asymptotic factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class Config:
    m: int = 24
    n: int = 40
    p: float = 1.0
    rho: float = 0.2
    num_instances: int = 20
    base_seed: int = 20260710
    sigma_upper: float = 2.0
    burn_in: int = 4000
    tail_length: int = 4000
    trace_iterations: int = 120
    s_min: float = 0.8
    s_max: float = 1.2
    h_row_min: float = 0.35
    h_row_max: float = 0.55
    h_null_min: float = 0.8
    h_null_max: float = 1.2


SIGMA_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def orthogonal_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def modal_roots(
    sigma: float,
    s_value: float,
    h_value: float,
    p: float,
    rho: float,
) -> np.ndarray:
    s_sq = s_value * s_value
    d_value = p + h_value + sigma * s_sq
    return np.roots(
        [d_value, -(d_value + p - 2.0 * rho * s_sq), p - rho * s_sq]
    )


def modal_radius(
    sigma: float,
    s_values: np.ndarray,
    h_row: np.ndarray,
    h_null: np.ndarray,
    p: float,
    rho: float,
) -> float:
    row_radii = [
        float(np.max(np.abs(modal_roots(sigma, s, h, p, rho))))
        for s, h in zip(s_values, h_row)
    ]
    null_radii = list(p / (p + h_null))
    return max(row_radii + null_radii)


def individual_critical_sigmas(
    s_values: np.ndarray,
    h_row: np.ndarray,
    p: float,
    rho: float,
) -> np.ndarray:
    s_sq = s_values * s_values
    numerator = 2.0 * np.sqrt(rho * s_sq * (p - rho * s_sq)) - h_row
    return np.maximum(numerator, 0.0) / s_sq


def predict_sigma(
    s_values: np.ndarray,
    h_row: np.ndarray,
    h_null: np.ndarray,
    config: Config,
) -> tuple[float, float]:
    objective = lambda sigma: modal_radius(
        sigma, s_values, h_row, h_null, config.p, config.rho
    )
    result = minimize_scalar(
        objective,
        bounds=(0.0, config.sigma_upper),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Modal sigma prediction failed: {result.message}")
    if result.x > 0.99 * config.sigma_upper:
        raise RuntimeError("Predicted sigma is too close to the preregistered upper bound")
    return float(result.x), float(result.fun)


def construct_instance(config: Config, index: int) -> dict[str, np.ndarray | float | int]:
    seed = config.base_seed + 1009 * index
    rng = np.random.default_rng(seed)
    s_values = np.sort(rng.uniform(config.s_min, config.s_max, size=config.m))
    h_row = rng.uniform(config.h_row_min, config.h_row_max, size=config.m)
    h_null = rng.uniform(config.h_null_min, config.h_null_max, size=config.n - config.m)

    u_matrix = orthogonal_matrix(rng, config.m)
    v_matrix = orthogonal_matrix(rng, config.n)
    rectangular_s = np.zeros((config.m, config.n))
    rectangular_s[np.arange(config.m), np.arange(config.m)] = s_values
    a_matrix = u_matrix @ rectangular_s @ v_matrix.T
    h_matrix = v_matrix @ np.diag(np.concatenate([h_row, h_null])) @ v_matrix.T
    h_matrix = 0.5 * (h_matrix + h_matrix.T)

    ata = a_matrix.T @ a_matrix
    commutator = h_matrix @ ata - ata @ h_matrix
    commutator_relative = np.linalg.norm(commutator, ord="fro") / max(
        np.linalg.norm(h_matrix, ord="fro") * np.linalg.norm(ata, ord="fro"),
        np.finfo(float).tiny,
    )
    metric_margin = config.p - config.rho * float(np.linalg.norm(a_matrix, ord=2) ** 2)
    if commutator_relative > 1e-12:
        raise RuntimeError(f"Simultaneous diagonalization check failed: {commutator_relative}")
    if metric_margin <= 0.0:
        raise RuntimeError(f"Metric condition failed: margin={metric_margin}")

    critical = individual_critical_sigmas(
        s_values, h_row, config.p, config.rho
    )
    predicted_sigma, predicted_radius = predict_sigma(
        s_values, h_row, h_null, config
    )
    return {
        "index": index,
        "seed": seed,
        "A": a_matrix,
        "H": h_matrix,
        "s_values": s_values,
        "h_row": h_row,
        "h_null": h_null,
        "critical": critical,
        "predicted_sigma": predicted_sigma,
        "predicted_radius": predicted_radius,
        "commutator_relative": commutator_relative,
        "metric_margin": metric_margin,
    }


def exact_map_matrix(
    a_matrix: np.ndarray,
    h_matrix: np.ndarray,
    sigma: float,
    config: Config,
) -> np.ndarray:
    n, m = config.n, config.m
    primal_system = (
        h_matrix
        + config.p * np.eye(n)
        + sigma * (a_matrix.T @ a_matrix)
    )
    x_from_x = np.linalg.solve(primal_system, config.p * np.eye(n))
    x_from_y = np.linalg.solve(primal_system, a_matrix.T)
    y_from_x = config.rho * a_matrix - 2.0 * config.rho * a_matrix @ x_from_x
    y_from_y = np.eye(m) - 2.0 * config.rho * a_matrix @ x_from_y
    return np.block([[x_from_x, x_from_y], [y_from_x, y_from_y]])


def metric_and_kkt(
    a_matrix: np.ndarray,
    h_matrix: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    metric = np.block(
        [
            [config.p * np.eye(config.n), a_matrix.T],
            [a_matrix, (1.0 / config.rho) * np.eye(config.m)],
        ]
    )
    kkt = np.block(
        [
            [h_matrix, -a_matrix.T],
            [a_matrix, np.zeros((config.m, config.m))],
        ]
    )
    return metric, kkt


def metric_norm(vector: np.ndarray, metric: np.ndarray) -> float:
    value = float(vector @ metric @ vector)
    return math.sqrt(max(value, 0.0))


def observed_factors(
    map_matrix: np.ndarray,
    metric: np.ndarray,
    kkt: np.ndarray,
    initial: np.ndarray,
    config: Config,
) -> tuple[float, float]:
    state = initial.copy()
    state /= metric_norm(state, metric)
    previous_kkt_norm = float(np.linalg.norm(kkt @ state))
    metric_logs: list[float] = []
    kkt_logs: list[float] = []
    total_iterations = config.burn_in + config.tail_length

    for iteration in range(total_iterations):
        raw_state = map_matrix @ state
        factor = metric_norm(raw_state, metric)
        if not math.isfinite(factor) or factor <= 0.0:
            raise RuntimeError(f"Invalid metric factor at iteration {iteration}: {factor}")
        state = raw_state / factor
        current_kkt_norm = float(np.linalg.norm(kkt @ state))
        kkt_factor = factor * current_kkt_norm / previous_kkt_norm
        if iteration >= config.burn_in:
            metric_logs.append(math.log(factor))
            kkt_logs.append(math.log(kkt_factor))
        previous_kkt_norm = current_kkt_norm

    return math.exp(float(np.mean(metric_logs))), math.exp(float(np.mean(kkt_logs)))


def residual_trace(
    map_matrix: np.ndarray,
    kkt: np.ndarray,
    initial: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = initial.copy()
    initial_residual = float(np.linalg.norm(kkt @ state))
    residuals = [1.0]
    for _ in range(iterations):
        state = map_matrix @ state
        residuals.append(float(np.linalg.norm(kkt @ state)) / initial_residual)
    return np.arange(iterations + 1), np.asarray(residuals)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    output_dir: Path,
    aggregate_rows: list[dict[str, object]],
    trace_data: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), constrained_layout=True)

    positions = np.arange(len(aggregate_rows))
    labels = [f"{float(row['sigma_multiplier']):g}" for row in aggregate_rows]
    theory = np.asarray([float(row["theory_median"]) for row in aggregate_rows])
    observed = np.asarray([float(row["observed_kkt_median"]) for row in aggregate_rows])
    q1 = np.asarray([float(row["observed_kkt_q1"]) for row in aggregate_rows])
    q3 = np.asarray([float(row["observed_kkt_q3"]) for row in aggregate_rows])

    axes[0].plot(
        positions,
        theory,
        color="#4f5965",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=4.0,
        label="Modal prediction",
    )
    axes[0].errorbar(
        positions,
        observed,
        yerr=np.vstack([observed - q1, q3 - observed]),
        color="#087e8b",
        linewidth=1.4,
        marker="o",
        markersize=4.3,
        capsize=2.5,
        label="Exact FA-CP trajectory",
    )
    axes[0].axvline(3, color="#b23a48", linewidth=1.1, linestyle=":")
    axes[0].set_xticks(positions, labels)
    axes[0].set_xlabel(r"Penalty ratio $\sigma/\widehat\sigma$")
    axes[0].set_ylabel("Asymptotic contraction factor")
    axes[0].grid(True, color="#dedede", linewidth=0.6)
    axes[0].legend(frameon=True, edgecolor="#999999")
    axes[0].text(
        3.08,
        min(theory.min(), observed.min()) + 0.01,
        r"$\widehat\sigma$",
        color="#8d2633",
        ha="left",
        va="bottom",
    )

    trace_styles = {
        0.0: ("#4f5965", "--", r"$\sigma=0$"),
        1.0: ("#087e8b", "-", r"$\sigma=\widehat\sigma$"),
        4.0: ("#2a6fbb", "-.", r"$\sigma=4\widehat\sigma$"),
    }
    for multiplier, (iterations, residuals) in trace_data.items():
        color, linestyle, label = trace_styles[multiplier]
        axes[1].semilogy(
            iterations,
            residuals,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Relative KKT residual")
    axes[1].grid(True, which="both", color="#dedede", linewidth=0.6)
    axes[1].legend(frameon=True, edgecolor="#999999")

    fig.savefig(output_dir / "modal_sigma_controlled.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "modal_sigma_controlled.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def run(config: Config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config)
    config_payload["sigma_multipliers"] = list(SIGMA_MULTIPLIERS)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )

    run_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    theory_by_multiplier: dict[float, list[float]] = {x: [] for x in SIGMA_MULTIPLIERS}
    metric_by_multiplier: dict[float, list[float]] = {x: [] for x in SIGMA_MULTIPLIERS}
    kkt_by_multiplier: dict[float, list[float]] = {x: [] for x in SIGMA_MULTIPLIERS}
    best_multiplier_counts: dict[float, int] = {x: 0 for x in SIGMA_MULTIPLIERS}
    trace_data: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for index in range(config.num_instances):
        instance = construct_instance(config, index)
        a_matrix = np.asarray(instance["A"])
        h_matrix = np.asarray(instance["H"])
        s_values = np.asarray(instance["s_values"])
        h_row = np.asarray(instance["h_row"])
        h_null = np.asarray(instance["h_null"])
        predicted_sigma = float(instance["predicted_sigma"])
        metric, kkt = metric_and_kkt(a_matrix, h_matrix, config)
        rng = np.random.default_rng(int(instance["seed"]) + 7919)
        initial = rng.normal(size=config.n + config.m)
        initial /= metric_norm(initial, metric)

        instance_results: list[tuple[float, float]] = []
        max_dense_modal_gap = 0.0
        for multiplier in SIGMA_MULTIPLIERS:
            sigma = multiplier * predicted_sigma
            map_matrix = exact_map_matrix(a_matrix, h_matrix, sigma, config)
            dense_radius = float(np.max(np.abs(np.linalg.eigvals(map_matrix))))
            theory_radius = modal_radius(
                sigma, s_values, h_row, h_null, config.p, config.rho
            )
            max_dense_modal_gap = max(max_dense_modal_gap, abs(dense_radius - theory_radius))
            observed_metric, observed_kkt = observed_factors(
                map_matrix, metric, kkt, initial, config
            )
            run_rows.append(
                {
                    "instance_index": index,
                    "seed": int(instance["seed"]),
                    "predicted_sigma": predicted_sigma,
                    "sigma_multiplier": multiplier,
                    "sigma": sigma,
                    "theory_radius": theory_radius,
                    "dense_map_radius": dense_radius,
                    "observed_metric_factor": observed_metric,
                    "observed_kkt_factor": observed_kkt,
                    "dense_modal_abs_gap": abs(dense_radius - theory_radius),
                }
            )
            theory_by_multiplier[multiplier].append(theory_radius)
            metric_by_multiplier[multiplier].append(observed_metric)
            kkt_by_multiplier[multiplier].append(observed_kkt)
            instance_results.append((multiplier, observed_kkt))

            if index == 0 and multiplier in (0.0, 1.0, 4.0):
                trace_data[multiplier] = residual_trace(
                    map_matrix, kkt, initial, config.trace_iterations
                )

        best_multiplier = min(instance_results, key=lambda item: item[1])[0]
        best_multiplier_counts[best_multiplier] += 1
        critical = np.asarray(instance["critical"])
        instance_rows.append(
            {
                "instance_index": index,
                "seed": int(instance["seed"]),
                "predicted_sigma": predicted_sigma,
                "predicted_radius": float(instance["predicted_radius"]),
                "critical_sigma_min": float(np.min(critical)),
                "critical_sigma_median": float(np.median(critical)),
                "critical_sigma_max": float(np.max(critical)),
                "observed_best_multiplier": best_multiplier,
                "commutator_relative": float(instance["commutator_relative"]),
                "metric_margin": float(instance["metric_margin"]),
                "max_dense_modal_gap": max_dense_modal_gap,
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    for multiplier in SIGMA_MULTIPLIERS:
        theory_values = theory_by_multiplier[multiplier]
        metric_values = metric_by_multiplier[multiplier]
        kkt_values = kkt_by_multiplier[multiplier]
        aggregate_rows.append(
            {
                "sigma_multiplier": multiplier,
                "theory_median": float(np.median(theory_values)),
                "observed_metric_median": float(np.median(metric_values)),
                "observed_kkt_median": float(np.median(kkt_values)),
                "observed_kkt_q1": percentile(kkt_values, 25.0),
                "observed_kkt_q3": percentile(kkt_values, 75.0),
                "instances_best": best_multiplier_counts[multiplier],
            }
        )

    write_csv(output_dir / "modal_sigma_runs.csv", run_rows)
    write_csv(output_dir / "modal_sigma_instances.csv", instance_rows)
    write_csv(output_dir / "modal_sigma_aggregate.csv", aggregate_rows)
    make_figure(output_dir, aggregate_rows, trace_data)

    max_modal_gap = max(float(row["dense_modal_abs_gap"]) for row in run_rows)
    max_observed_gap = max(
        abs(float(row["observed_kkt_factor"]) - float(row["theory_radius"]))
        for row in run_rows
    )
    summary = {
        "num_instances": config.num_instances,
        "predicted_sigma_min": min(float(row["predicted_sigma"]) for row in instance_rows),
        "predicted_sigma_median": float(
            np.median([float(row["predicted_sigma"]) for row in instance_rows])
        ),
        "predicted_sigma_max": max(float(row["predicted_sigma"]) for row in instance_rows),
        "instances_best_at_prediction": best_multiplier_counts[1.0],
        "max_dense_modal_gap": max_modal_gap,
        "max_observed_theory_gap": max_observed_gap,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("sigma/pred  theory_med  observed_KKT_med  IQR  best_count")
    for row in aggregate_rows:
        print(
            f"{float(row['sigma_multiplier']):>10g}  "
            f"{float(row['theory_median']):.9f}  "
            f"{float(row['observed_kkt_median']):.9f}  "
            f"[{float(row['observed_kkt_q1']):.9f}, {float(row['observed_kkt_q3']):.9f}]  "
            f"{int(row['instances_best']):>2d}/{config.num_instances}"
        )
    print(json.dumps(summary, indent=2))
    print(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "modal_sigma_controlled_v1",
    )
    parser.add_argument("--num-instances", type=int, default=Config.num_instances)
    parser.add_argument("--burn-in", type=int, default=Config.burn_in)
    parser.add_argument("--tail-length", type=int, default=Config.tail_length)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        num_instances=args.num_instances,
        burn_in=args.burn_in,
        tail_length=args.tail_length,
    )
    run(config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
