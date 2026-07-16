import csv
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "qp_linearized" / "collect_qp_linearized.py"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_native_summary(path, runtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Termination Reason: OPTIMAL\nRuntime (sec): {runtime:.8e}\n",
        encoding="utf-8",
    )


def base_row(instance, elapsed, iterations):
    return {
        "instance": instance,
        "instance_name": instance,
        "family_key": "family_a",
        "h_family": "diagonal",
        "box_regime": "interior",
        "size_class": "small",
        "status": "ok",
        "optimal": "1",
        "termination_reason": "OPTIMAL",
        "elapsed_sec": str(elapsed),
        "total_iters": str(iterations),
        "rel_primal": "1e-7",
        "rel_dual": "2e-7",
        "rel_gap": "3e-7",
    }


def test_collector_combines_complete_fixed_profiles(tmp_path):
    exact = []
    for key in ("pdhcg", "cpal", "facp"):
        for index, instance in enumerate(("a", "b"), start=1):
            row = base_row(instance, index, 100 * index)
            row["method_key"] = key
            exact.append(row)
    joined = tmp_path / "original_run" / "analysis" / "joined.csv"
    write_csv(joined, exact)

    pdhg_root = tmp_path / "protocol_run" / "results" / "lin_pdhg_ref0"
    cpal_root = tmp_path / "cpal_run" / "results" / "lin_cpal_profile"
    write_csv(
        pdhg_root / "chunk_00" / "summary.csv",
        [base_row("a", 3, 100), base_row("b", 4, 200)],
    )
    write_csv(
        cpal_root / "chunk_00" / "summary.csv",
        [base_row("a", 2, 100), base_row("b", 3, 200)],
    )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--exact-joined",
            str(joined),
            "--lin-pdhg-root",
            str(pdhg_root),
            "--lin-cpal-root",
            str(cpal_root),
            "--output-dir",
            str(output),
            "--expected-count",
            "2",
        ],
        check=True,
    )

    with (output / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method_key"] for row in rows] == [
        "pdhcg",
        "cpal",
        "facp",
        "lin_pdhg",
        "lin_cpal",
    ]
    assert float(rows[-1]["total_elapsed"]) == 5.0
    assert rows[-1]["source_run"] == "cpal_run"


def test_collector_retains_native_runtime_and_aggregates_wrapper_time(tmp_path):
    exact = []
    exact_root = tmp_path / "original_run"
    for method_index, key in enumerate(("pdhcg", "cpal", "facp"), start=1):
        for instance_index, instance in enumerate(("a", "b"), start=1):
            row = base_row(instance, 99, 100 * instance_index)
            row["method_key"] = key
            exact.append(row)
            write_native_summary(
                exact_root
                / "family_a"
                / key
                / "default"
                / "solver_outputs"
                / instance
                / f"{instance}_summary.txt",
                method_index + instance_index / 10,
            )
    joined = exact_root / "analysis" / "joined.csv"
    write_csv(joined, exact)

    pdhg_root = tmp_path / "protocol_run" / "results" / "lin_pdhg_ref0"
    cpal_root = tmp_path / "cpal_run" / "results" / "lin_cpal_profile"
    for root, base in ((pdhg_root, 4.0), (cpal_root, 5.0)):
        rows = [base_row("a", 99, 100), base_row("b", 99, 200)]
        write_csv(root / "chunk_00" / "summary.csv", rows)
        for instance_index, instance in enumerate(("a", "b"), start=1):
            write_native_summary(
                root
                / "chunk_00"
                / "solver_outputs"
                / instance
                / f"{instance}_summary.txt",
                base + instance_index / 10,
            )

    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--exact-joined",
            str(joined),
            "--exact-native-root",
            str(exact_root),
            "--lin-pdhg-root",
            str(pdhg_root),
            "--lin-cpal-root",
            str(cpal_root),
            "--output-dir",
            str(output),
            "--expected-count",
            "2",
        ],
        check=True,
    )

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    with (output / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        aggregates = list(csv.DictReader(handle))
    with (output / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    assert audit["timing_source"] == "per_invocation_wrapper"
    assert audit["display_elapsed_field"] == "wrapper_elapsed_sec"
    assert set(audit["native_runtime_summaries"]) == {
        "pdhcg",
        "cpal",
        "facp",
        "lin_pdhg",
        "lin_cpal",
    }
    assert all(
        value["summary_count"] == 2
        and len(value["tree_sha256"]) == 64
        and not Path(value["source_run"]).is_absolute()
        for value in audit["native_runtime_summaries"].values()
    )
    assert math.isclose(float(aggregates[-1]["total_elapsed"]), 198.0)
    assert all(float(row["wrapper_elapsed_sec"]) == 99.0 for row in runs)
    assert any(float(row["elapsed_sec"]) != 99.0 for row in runs)


def test_collector_accepts_the_fresh_five_profile_runner_layout(tmp_path):
    manifest = tmp_path / "dataset" / "manifest.csv"
    write_csv(
        manifest,
        [
            {
                "instance_name": instance,
                "family_key": "family_a",
                "h_family": "diagonal",
                "box_regime": "interior",
                "size_class": "small",
            }
            for instance in ("a", "b")
        ],
    )
    run_root = tmp_path / "fresh_run" / "results"
    profiles = {
        "pdhcg_original": 1.0,
        "rhr_cpal_subproblem": 2.0,
        "rhr_facp_subproblem": 3.0,
        "rhr_lin_pdhg": 4.0,
        "rhr_lin_cpal": 5.0,
    }
    for profile, base in profiles.items():
        root = run_root / profile / "chunk_00"
        write_csv(
            root / "summary.csv",
            [base_row("a", 99, 100), base_row("b", 99, 200)],
        )
        for offset, instance in enumerate(("a", "b"), start=1):
            write_native_summary(
                root / "solver_outputs" / instance / f"{instance}_summary.txt",
                base + offset / 10,
            )

    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--expected-count",
            "2",
        ],
        check=True,
    )

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    with (output / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        aggregates = list(csv.DictReader(handle))
    assert audit["timing_source"] == "per_invocation_wrapper"
    assert set(audit["input_artifacts"]["profile_chunk_summaries"]) == {
        "pdhcg",
        "cpal",
        "facp",
        "lin_pdhg",
        "lin_cpal",
    }
    assert [row["method_key"] for row in aggregates] == [
        "pdhcg",
        "cpal",
        "facp",
        "lin_pdhg",
        "lin_cpal",
    ]
    assert math.isclose(float(aggregates[-1]["total_elapsed"]), 198.0)
