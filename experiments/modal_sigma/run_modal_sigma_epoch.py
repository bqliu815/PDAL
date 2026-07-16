#!/usr/bin/env python3
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Validate the restarted reflected-Halpern epoch penalty prediction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


BASE_SCRIPT = Path(__file__).resolve().with_name("run_modal_sigma_controlled.py")
SPEC = importlib.util.spec_from_file_location("modal_sigma_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import base experiment: {BASE_SCRIPT}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


EPOCH_LENGTHS = (1, 3, 5, 10, 20)
GAMMA = 1.0
GRID_SIZE = 2001
OBSERVED_BURN_IN = 2000
OBSERVED_TAIL = 2000


def q_value(value: np.ndarray | complex, epoch_length: int) -> np.ndarray | complex:
    result = np.ones_like(value, dtype=np.complex128)
    power = np.ones_like(value, dtype=np.complex128)
    for _ in range(epoch_length):
        power = power * value
        result = result + power
    return result / (epoch_length + 1)


def modal_eigenvalues(
    sigma: float,
    instance: dict[str, object],
    config: object,
) -> np.ndarray:
    s_values = np.asarray(instance["s_values"])
    h_row = np.asarray(instance["h_row"])
    h_null = np.asarray(instance["h_null"])
    s_sq = s_values * s_values
    d_value = config.p + h_row + sigma * s_sq
    trace = (d_value + config.p - 2.0 * config.rho * s_sq) / d_value
    determinant = (config.p - config.rho * s_sq) / d_value
    discriminant = trace * trace - 4.0 * determinant
    square_root = np.sqrt(discriminant.astype(np.complex128))
    roots = np.concatenate([(trace + square_root) / 2.0, (trace - square_root) / 2.0])
    null_roots = config.p / (config.p + h_null)
    return np.concatenate([roots, null_roots.astype(np.complex128)])


def epoch_modal_radius(
    sigma: float,
    instance: dict[str, object],
    config: object,
    epoch_length: int,
) -> float:
    eigenvalues = modal_eigenvalues(sigma, instance, config)
    reflected = (1.0 + GAMMA) * eigenvalues - GAMMA
    return float(np.max(np.abs(q_value(reflected, epoch_length))))


def predict_epoch_sigma(
    instance: dict[str, object],
    config: object,
    epoch_length: int,
) -> tuple[float, float]:
    grid = np.linspace(0.0, config.sigma_upper, GRID_SIZE)
    values = np.asarray(
        [epoch_modal_radius(x, instance, config, epoch_length) for x in grid]
    )
    index = int(np.argmin(values))
    if index == 0:
        return 0.0, float(values[0])
    left = float(grid[index - 1])
    right = float(grid[min(index + 1, GRID_SIZE - 1)])
    result = minimize_scalar(
        lambda sigma: epoch_modal_radius(sigma, instance, config, epoch_length),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Epoch prediction failed: {result.message}")
    return float(result.x), float(result.fun)


def epoch_matrix(map_matrix: np.ndarray, epoch_length: int) -> np.ndarray:
    identity = np.eye(map_matrix.shape[0])
    reflected = (1.0 + GAMMA) * map_matrix - GAMMA * identity
    current = identity.copy()
    for index in range(epoch_length):
        current = (identity + (index + 1.0) * reflected @ current) / (index + 2.0)
    return current


def observed_epoch_factor(
    epoch_map: np.ndarray,
    metric: np.ndarray,
    kkt: np.ndarray,
    initial: np.ndarray,
) -> tuple[float, float]:
    state = initial.copy()
    state /= BASE.metric_norm(state, metric)
    previous_kkt = float(np.linalg.norm(kkt @ state))
    metric_logs: list[float] = []
    kkt_logs: list[float] = []
    for iteration in range(OBSERVED_BURN_IN + OBSERVED_TAIL):
        raw_state = epoch_map @ state
        factor = BASE.metric_norm(raw_state, metric)
        if not math.isfinite(factor) or factor <= 0.0:
            raise RuntimeError(f"Invalid epoch factor at {iteration}: {factor}")
        state = raw_state / factor
        current_kkt = float(np.linalg.norm(kkt @ state))
        kkt_factor = factor * current_kkt / previous_kkt
        if iteration >= OBSERVED_BURN_IN:
            metric_logs.append(math.log(factor))
            kkt_logs.append(math.log(kkt_factor))
        previous_kkt = current_kkt
    return math.exp(float(np.mean(metric_logs))), math.exp(float(np.mean(kkt_logs)))


def append_candidate(
    candidates: list[tuple[str, float]],
    tag: str,
    sigma: float,
) -> None:
    for _, old_sigma in candidates:
        if math.isclose(old_sigma, sigma, rel_tol=1e-6, abs_tol=1e-9):
            return
    candidates.append((tag, sigma))


def candidates_for(epoch_sigma: float, shadow_sigma: float) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    append_candidate(candidates, "zero", 0.0)
    append_candidate(candidates, "epoch_pred", epoch_sigma)
    append_candidate(candidates, "quarter_shadow", 0.25 * shadow_sigma)
    append_candidate(candidates, "half_shadow", 0.5 * shadow_sigma)
    append_candidate(candidates, "shadow_pred", shadow_sigma)
    append_candidate(candidates, "double_shadow", 2.0 * shadow_sigma)
    append_candidate(candidates, "four_shadow", 4.0 * shadow_sigma)
    return candidates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(output_dir: Path, aggregate_rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
        }
    )
    epoch_lengths = np.asarray([int(row["epoch_length"]) for row in aggregate_rows])
    sigma_ratio = np.asarray([float(row["epoch_to_shadow_sigma_median"]) for row in aggregate_rows])
    sigma_q1 = np.asarray([float(row["epoch_to_shadow_sigma_q1"]) for row in aggregate_rows])
    sigma_q3 = np.asarray([float(row["epoch_to_shadow_sigma_q3"]) for row in aggregate_rows])
    predicted = np.asarray([float(row["observed_predicted_median"]) for row in aggregate_rows])
    zero = np.asarray([float(row["observed_zero_median"]) for row in aggregate_rows])
    shadow = np.asarray([float(row["observed_shadow_median"]) for row in aggregate_rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    positions = np.arange(len(epoch_lengths))
    axes[0].errorbar(
        positions,
        sigma_ratio,
        yerr=np.vstack([sigma_ratio - sigma_q1, sigma_q3 - sigma_ratio]),
        color="#087e8b",
        marker="o",
        linewidth=1.7,
        capsize=2.5,
    )
    axes[0].axhline(0.0, color="#4f5965", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(positions, [str(x) for x in epoch_lengths])
    axes[0].set_xlabel(r"Epoch length $K$")
    axes[0].set_ylabel(r"Predicted ratio $\widehat\sigma_K/\widehat\sigma_1$")
    axes[0].grid(True, color="#dedede", linewidth=0.6)

    axes[1].plot(
        positions,
        zero,
        color="#4f5965",
        linestyle="--",
        marker="s",
        linewidth=1.5,
        label=r"$\sigma=0$",
    )
    axes[1].plot(
        positions,
        shadow,
        color="#2a6fbb",
        linestyle="-.",
        marker="^",
        linewidth=1.5,
        label=r"Shadow-map $\widehat\sigma_1$",
    )
    axes[1].plot(
        positions,
        predicted,
        color="#087e8b",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        label=r"Epoch prediction $\widehat\sigma_K$",
    )
    axes[1].set_xticks(positions, [str(x) for x in epoch_lengths])
    axes[1].set_xlabel(r"Epoch length $K$")
    axes[1].set_ylabel("Observed epoch contraction factor")
    axes[1].grid(True, color="#dedede", linewidth=0.6)
    axes[1].legend(frameon=True, edgecolor="#999999")

    fig.savefig(output_dir / "modal_sigma_epoch.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "modal_sigma_epoch.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BASE.Config()
    config_payload = {
        "base_config": BASE.asdict(config),
        "epoch_lengths": list(EPOCH_LENGTHS),
        "gamma": GAMMA,
        "sigma_grid_size": GRID_SIZE,
        "observed_burn_in_epochs": OBSERVED_BURN_IN,
        "observed_tail_epochs": OBSERVED_TAIL,
        "candidate_rule": [
            "zero",
            "epoch_pred",
            "0.25*shadow_pred",
            "0.5*shadow_pred",
            "shadow_pred",
            "2*shadow_pred",
            "4*shadow_pred",
        ],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )

    run_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    for instance_index in range(config.num_instances):
        instance = BASE.construct_instance(config, instance_index)
        a_matrix = np.asarray(instance["A"])
        h_matrix = np.asarray(instance["H"])
        shadow_sigma = float(instance["predicted_sigma"])
        metric, kkt = BASE.metric_and_kkt(a_matrix, h_matrix, config)
        rng = np.random.default_rng(int(instance["seed"]) + 104729)
        initial = rng.normal(size=config.n + config.m)
        initial /= BASE.metric_norm(initial, metric)

        for epoch_length in EPOCH_LENGTHS:
            epoch_sigma, predicted_radius = predict_epoch_sigma(
                instance, config, epoch_length
            )
            candidate_results: list[tuple[str, float, float]] = []
            max_formula_gap = 0.0
            for tag, sigma in candidates_for(epoch_sigma, shadow_sigma):
                shadow_map = BASE.exact_map_matrix(a_matrix, h_matrix, sigma, config)
                exact_epoch = epoch_matrix(shadow_map, epoch_length)
                dense_radius = float(np.max(np.abs(np.linalg.eigvals(exact_epoch))))
                theory_radius = epoch_modal_radius(
                    sigma, instance, config, epoch_length
                )
                max_formula_gap = max(max_formula_gap, abs(dense_radius - theory_radius))
                observed_metric, observed_kkt = observed_epoch_factor(
                    exact_epoch, metric, kkt, initial
                )
                run_rows.append(
                    {
                        "instance_index": instance_index,
                        "seed": int(instance["seed"]),
                        "epoch_length": epoch_length,
                        "gamma": GAMMA,
                        "shadow_predicted_sigma": shadow_sigma,
                        "epoch_predicted_sigma": epoch_sigma,
                        "candidate_tag": tag,
                        "sigma": sigma,
                        "sigma_over_shadow_prediction": sigma / shadow_sigma,
                        "theory_epoch_radius": theory_radius,
                        "dense_epoch_radius": dense_radius,
                        "observed_metric_factor": observed_metric,
                        "observed_kkt_factor": observed_kkt,
                        "dense_modal_abs_gap": abs(dense_radius - theory_radius),
                    }
                )
                candidate_results.append((tag, sigma, observed_kkt))

            best_tag, best_sigma, _ = min(candidate_results, key=lambda item: item[2])
            expected_tag = "zero" if epoch_sigma == 0.0 else "epoch_pred"
            prediction_wins = int(
                math.isclose(
                    best_sigma,
                    epoch_sigma,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            )
            instance_rows.append(
                {
                    "instance_index": instance_index,
                    "seed": int(instance["seed"]),
                    "epoch_length": epoch_length,
                    "shadow_predicted_sigma": shadow_sigma,
                    "epoch_predicted_sigma": epoch_sigma,
                    "epoch_to_shadow_sigma": epoch_sigma / shadow_sigma,
                    "predicted_epoch_radius": predicted_radius,
                    "expected_best_tag": expected_tag,
                    "observed_best_tag": best_tag,
                    "observed_best_sigma": best_sigma,
                    "prediction_wins": prediction_wins,
                    "max_dense_modal_gap": max_formula_gap,
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for epoch_length in EPOCH_LENGTHS:
        rows = [row for row in instance_rows if int(row["epoch_length"]) == epoch_length]
        run_subset = [row for row in run_rows if int(row["epoch_length"]) == epoch_length]

        predicted_values: list[float] = []
        zero_values: list[float] = []
        shadow_values: list[float] = []
        for instance_index in range(config.num_instances):
            candidate_rows = [
                row
                for row in run_subset
                if int(row["instance_index"]) == instance_index
            ]
            epoch_sigma = float(
                next(
                    row["epoch_predicted_sigma"]
                    for row in rows
                    if int(row["instance_index"]) == instance_index
                )
            )
            target_tag = "zero" if epoch_sigma == 0.0 else "epoch_pred"
            predicted_values.append(
                float(
                    next(
                        row["observed_kkt_factor"]
                        for row in candidate_rows
                    if row["candidate_tag"] == target_tag
                    )
                )
            )

            shadow_sigma = float(
                next(
                    row["shadow_predicted_sigma"]
                    for row in rows
                    if int(row["instance_index"]) == instance_index
                )
            )
            zero_row = min(candidate_rows, key=lambda row: abs(float(row["sigma"])))
            shadow_row = min(
                candidate_rows,
                key=lambda row: abs(float(row["sigma"]) - shadow_sigma),
            )
            zero_values.append(float(zero_row["observed_kkt_factor"]))
            shadow_values.append(float(shadow_row["observed_kkt_factor"]))

        ratios = [float(row["epoch_to_shadow_sigma"]) for row in rows]
        aggregate_rows.append(
            {
                "epoch_length": epoch_length,
                "positive_prediction_count": sum(
                    float(row["epoch_predicted_sigma"]) > 0.0 for row in rows
                ),
                "epoch_to_shadow_sigma_median": float(np.median(ratios)),
                "epoch_to_shadow_sigma_q1": float(np.percentile(ratios, 25.0)),
                "epoch_to_shadow_sigma_q3": float(np.percentile(ratios, 75.0)),
                "observed_predicted_median": float(np.median(predicted_values)),
                "observed_zero_median": float(np.median(zero_values)),
                "observed_shadow_median": float(np.median(shadow_values)),
                "prediction_wins": sum(int(row["prediction_wins"]) for row in rows),
                "num_instances": config.num_instances,
            }
        )

    write_csv(output_dir / "epoch_runs.csv", run_rows)
    write_csv(output_dir / "epoch_instances.csv", instance_rows)
    write_csv(output_dir / "epoch_aggregate.csv", aggregate_rows)
    make_figure(output_dir, aggregate_rows)

    summary = {
        "num_instances": config.num_instances,
        "epoch_lengths": list(EPOCH_LENGTHS),
        "gamma": GAMMA,
        "prediction_wins_total": sum(int(row["prediction_wins"]) for row in instance_rows),
        "comparisons_total": len(instance_rows),
        "max_dense_modal_gap": max(float(row["dense_modal_abs_gap"]) for row in run_rows),
        "max_observed_theory_gap": max(
            abs(float(row["observed_kkt_factor"]) - float(row["theory_epoch_radius"]))
            for row in run_rows
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("K  positive_pred  sigmaK/sigma1  pred_obs  zero_obs  shadow_obs  wins")
    for row in aggregate_rows:
        print(
            f"{int(row['epoch_length']):>2d}  "
            f"{int(row['positive_prediction_count']):>2d}/{config.num_instances}  "
            f"{float(row['epoch_to_shadow_sigma_median']):.6f}  "
            f"{float(row['observed_predicted_median']):.9f}  "
            f"{float(row['observed_zero_median']):.9f}  "
            f"{float(row['observed_shadow_median']):.9f}  "
            f"{int(row['prediction_wins']):>2d}/{config.num_instances}"
        )
    print(json.dumps(summary, indent=2))
    print(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "modal_sigma_epoch_v1",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
