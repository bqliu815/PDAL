from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "experiments" / "lp_benchmarks" / "collect_lp_tables.py"
SPEC = importlib.util.spec_from_file_location("lp_table_collector", COLLECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_sgm_charges_unsolved_run_at_manifest_limit() -> None:
    rows = [
        {
            "dataset": "MIPLIB",
            "split": "Small",
            "tolerance": "1e-4",
            "status": "OPTIMAL",
            "time_sec": "1",
            "time_limit": "100",
            "rel_primal": "1e-5",
            "rel_dual": "1e-5",
            "rel_gap": "1e-5",
        },
        {
            "dataset": "MIPLIB",
            "split": "Small",
            "tolerance": "1e-4",
            "status": "TIME_LIMIT",
            "time_sec": "99",
            "time_limit": "100",
            "rel_primal": "1",
            "rel_dual": "1",
            "rel_gap": "1",
        },
    ]
    result = COLLECTOR.summarize(rows, "MIPLIB", "Small", "1e-4")
    expected = math.sqrt((1.0 + 10.0) * (100.0 + 10.0)) - 10.0
    assert result["count"] == 2
    assert result["solved"] == 1
    assert math.isclose(result["sgm10"], expected)


def test_optimal_row_reported_after_time_limit_is_charged_as_unsolved() -> None:
    row = {
        "dataset": "MIPLIB",
        "split": "Small",
        "tolerance": "1e-4",
        "status": "OPTIMAL",
        "time_sec": "102",
        "time_limit": "100",
        "rel_primal": "1e-5",
        "rel_dual": "1e-5",
        "rel_gap": "1e-5",
    }
    result = COLLECTOR.summarize([row], "MIPLIB", "Small", "1e-4")
    assert result["count"] == 1
    assert result["solved"] == 0
    assert math.isclose(result["sgm10"], 100.0)

    row["time_sec"] = "100.5"
    result = COLLECTOR.summarize([row], "MIPLIB", "Small", "1e-4")
    assert result["solved"] == 1
    assert math.isclose(result["sgm10"], 100.0)


def test_completeness_audit_requires_every_method() -> None:
    manifest = [
        {
            "dataset": "MIPLIB",
            "split": "Small",
            "instance": "demo",
            "tolerance": "1e-4",
        }
    ]
    rows = [
        {
            "solver_id": method,
            "dataset": "MIPLIB",
            "split": "Small",
            "instance": "demo",
            "tolerance": "1e-4",
            "source": "fixture",
        }
        for method in COLLECTOR.METHOD_ORDER
    ]
    assert COLLECTOR.validate_rows(manifest, rows) == []
    errors = COLLECTOR.validate_rows(manifest, rows[:-1])
    assert len(errors) == 1
    assert "missing row" in errors[0]


def test_cpal_policy_log_records_positive_augmentation(tmp_path: Path) -> None:
    log = tmp_path / "solver.log"
    log.write_text(
        "[unified-policy] version=2 route=high_equality_tail_or_base\n"
        "[alpha-direct-pulse-start] charged_iter=10000 alpha=2e-3\n"
        "[alpha-direct-pulse-final] charged_iter=10200 "
        "positive_charged_iters=200 final_alpha=0\n",
        encoding="utf-8",
    )
    route, iterations = COLLECTOR.parse_cpal_policy_log(log)
    assert route == "high_equality_tail_or_base"
    assert iterations == 200


def test_cpal_wrapper_requires_successful_exit(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text(
        '{"wall_seconds": 2.5, "exit_code": 0}\n', encoding="utf-8"
    )
    assert COLLECTOR.parse_wrapper(wrapper) == (2.5, 0)
    wrapper.write_text(
        '{"wall_seconds": 2.5, "exit_code": 1}\n', encoding="utf-8"
    )
    try:
        COLLECTOR.parse_wrapper(wrapper)
    except RuntimeError as exc:
        assert "solver exit code 1" in str(exc)
    else:
        raise AssertionError("nonzero CP-AL exit code was accepted")


def test_cpal_summary_resolver_matches_executable_dotted_name_rule(
    tmp_path: Path,
) -> None:
    output = tmp_path / "task" / "solver_output"
    output.mkdir(parents=True)
    summary = output / "thk_summary.txt"
    summary.write_text("Termination Reason: OPTIMAL\n", encoding="utf-8")
    assert COLLECTOR.resolve_cpal_summary(tmp_path / "task", "thk.48") == summary


def test_cpal_native_summary_accepts_current_objective_key(tmp_path: Path) -> None:
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "Termination Reason: OPTIMAL\n"
        "Runtime (sec): 1.5\n"
        "Primal Objective Value: 2.75\n",
        encoding="utf-8",
    )
    values = COLLECTOR.parse_text_summary(summary)
    assert COLLECTOR.summary_value(
        values, "Primal Objective", "Primal Objective Value"
    ) == "2.75"


def test_latex_boldface_uses_count_then_time() -> None:
    group = [
        {"solved": 9, "sgm10": 1.0},
        {"solved": 10, "sgm10": 3.0},
        {"solved": 10, "sgm10": 2.0},
    ]
    assert COLLECTOR.latex_count_time(group[0], group) == ("9", "1.000")
    assert COLLECTOR.latex_count_time(group[1], group) == (
        r"\textbf{10}",
        "3.000",
    )
    assert COLLECTOR.latex_count_time(group[2], group) == (
        r"\textbf{10}",
        r"\textbf{2.000}",
    )


def test_cuda_initialization_failure_is_not_counted_as_solver_failure(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary_2.csv"
    instance_dir = tmp_path / "solver_outputs" / "demo"
    instance_dir.mkdir(parents=True)
    (instance_dir / "stderr.log").write_text(
        "CUDA context cannot be initialized\n",
        encoding="utf-8",
    )
    row = {"source": str(summary), "instance": "demo"}
    assert "CUDA context" in str(COLLECTOR.baseline_infrastructure_failure(row))


def test_portable_manifest_removes_machine_paths() -> None:
    rows = [
        {
            "dataset": "MIPLIB",
            "split": "Small",
            "instance": "demo",
            "tolerance": "1e-4",
            "time_limit": "3600",
            "mps": "/cluster/private/path/demo.mps",
        }
    ]
    assert COLLECTOR.portable_manifest(rows) == [
        {
            "task_index": 0,
            "dataset": "MIPLIB",
            "split": "Small",
            "instance": "demo",
            "tolerance": "1e-4",
            "time_limit": "3600",
            "input_file": "demo.mps",
        }
    ]


def test_input_hashes_collapse_the_two_tolerances(tmp_path: Path) -> None:
    matrix = tmp_path / "demo.mps"
    matrix.write_text("NAME DEMO\nENDATA\n", encoding="ascii")
    rows = [
        {
            "dataset": "MIPLIB",
            "split": "Small",
            "instance": "demo",
            "tolerance": tolerance,
            "mps": str(matrix),
        }
        for tolerance in ("1e-4", "1e-8")
    ]
    hashes = COLLECTOR.input_hash_rows(rows)
    assert len(hashes) == 1
    assert hashes[0]["input_file"] == "demo.mps"
    assert hashes[0]["sha256"] == COLLECTOR.sha256(matrix)


def test_software_provenance_is_normalized(tmp_path: Path) -> None:
    record = tmp_path / "software_provenance.txt"
    record.write_text(
        "2026-07-16T01:50:17+08:00\n"
        "julia version 1.11.2\n"
        + "a" * 64
        + "  /cluster/bin/cupdlpx\n"
        + "/cluster/src/cuPDLPx-C-v0.2.9\t"
        + "b" * 40
        + "\n",
        encoding="utf-8",
    )
    parsed = COLLECTOR.parse_software_provenance(record)
    assert parsed["metadata"] == ["2026-07-16T01:50:17+08:00", "julia version 1.11.2"]
    assert parsed["artifact_hashes"] == [{"artifact": "cupdlpx", "sha256": "a" * 64}]
    assert parsed["source_revisions"] == [
        {"repository": "cuPDLPx-C-v0.2.9", "revision": "b" * 40}
    ]


def test_execution_retry_record_accepts_only_frozen_cuda_retries(
    tmp_path: Path,
) -> None:
    record = tmp_path / "execution_retries.tsv"
    record.write_text(
        "job_id\tmethod\ttask_scope\ttask_indices\tcause\tdisposition\n"
        "123\trhr_cpal\tfull856_manifest\t10,11\t"
        "cuda_context_failure\tsame manifest, binary, and solver settings\n",
        encoding="utf-8",
    )
    rows = COLLECTOR.parse_execution_retries(record)
    assert rows[0]["job_id"] == "123"

    record.write_text(
        "job_id\tmethod\ttask_scope\ttask_indices\tcause\tdisposition\n"
        "124\trhr_cpal\tfull856_manifest\t12\t"
        "solver_timeout\tsame manifest, binary, and solver settings\n",
        encoding="utf-8",
    )
    try:
        COLLECTOR.parse_execution_retries(record)
    except RuntimeError as exc:
        assert "non-infrastructure retry" in str(exc)
    else:
        raise AssertionError("solver-level retry was accepted")


def test_formal_wrapper_hash_validation() -> None:
    assert COLLECTOR.validated_sha256("a" * 64, "formal wrapper") == "a" * 64
    try:
        COLLECTOR.validated_sha256("not-a-hash", "formal wrapper")
    except RuntimeError as exc:
        assert "expected a lowercase SHA256 digest" in str(exc)
    else:
        raise AssertionError("invalid formal wrapper hash was accepted")


def test_collector_normalizes_dotted_cupdlpx_output_name(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary_0.csv"
    instance_dir = tmp_path / "solver_outputs" / "thk.48"
    instance_dir.mkdir(parents=True)
    raw_summary = instance_dir / "thk_summary.txt"
    raw_summary.write_text(
        "Termination Reason: OPTIMAL\n"
        "Runtime (sec): 2.5\n"
        "Iterations Count: 20\n"
        "Primal Objective: 1.0\n"
        "Relative Objective Gap: 1e-9\n"
        "Relative Primal Residual: 2e-9\n"
        "Relative Dual Residual: 3e-9\n",
        encoding="utf-8",
    )
    row = {
        "solver_id": "cupdlpx_c",
        "instance": "thk.48",
        "status": "ERROR",
        "source": str(summary_csv),
        "error": "missing output",
    }
    COLLECTOR.reparse_baseline_raw_output(row)
    assert row["status"] == "OPTIMAL"
    assert row["time_sec"] == "2.5"
    assert row["rel_dual"] == "3e-9"
    assert row["error"] == ""


def test_collector_normalizes_cupdlpc_nonfinite_json(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary_0.csv"
    instance_dir = tmp_path / "solver_outputs" / "dlr2"
    instance_dir.mkdir(parents=True)
    (instance_dir / "summary.json").write_text(
        '{"terminationCode":"TIMELIMIT_OR_ITERLIMIT",'
        '"dSolvingTime":1000.0,"nIter":200,"dPrimalObj":0.0,'
        '"dRelDualityGap":1.0,"dRelPrimalFeas":2.0,"dRelDualFeas":inf}',
        encoding="utf-8",
    )
    row = {
        "solver_id": "cupdlpc",
        "instance": "dlr2",
        "status": "ERROR",
        "source": str(summary_csv),
        "error": "invalid JSON",
    }
    COLLECTOR.reparse_baseline_raw_output(row)
    assert row["status"] == "TIMELIMIT_OR_ITERLIMIT"
    assert row["rel_dual"] == "inf"
    assert row["error"] == ""


def test_released_rows_remove_cluster_roots(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    cpal = tmp_path / "cpal"
    rows = [
        {
            "solver_id": "cupdlpx_c",
            "source": str(baseline / "cell" / "summary_0.csv"),
            "source_output": str(baseline / "cell" / "solver_outputs" / "demo.txt"),
            "error": f"missing {baseline}/cell/solver_outputs/demo.json",
        },
        {
            "solver_id": "rhr_cpal",
            "source": str(cpal / "task_000" / "summary.txt"),
            "source_output": "",
        },
    ]
    normalized = COLLECTOR.released_rows(rows, baseline, cpal)
    assert normalized[0]["source"] == "cell/summary_0.csv"
    assert normalized[0]["source_output"] == "cell/solver_outputs/demo.txt"
    assert str(baseline) not in normalized[0]["error"]
    assert normalized[1]["source"] == "task_000/summary.txt"
