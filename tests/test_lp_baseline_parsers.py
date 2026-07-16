from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "experiments" / "lp_benchmarks" / "parse_outputs.py"
RUNNER_PATH = ROOT / "experiments" / "lp_benchmarks" / "run_baseline_group.sbatch"
SPEC = importlib.util.spec_from_file_location("lp_baseline_parsers", PARSER_PATH)
assert SPEC is not None and SPEC.loader is not None
PARSERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSERS)


def test_cupdlpc_uses_relative_gap(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    output.write_text(
        json.dumps(
            {
                "terminationCode": "OPTIMAL",
                "dSolvingTime": 0.2,
                "nIter": 440,
                "dPrimalObj": -120.0,
                "dDualityGap": 0.035,
                "dRelDualityGap": 2.9e-5,
                "dRelPrimalFeas": 1.7e-5,
                "dRelDualFeas": 0.0,
            }
        ),
        encoding="utf-8",
    )
    row = PARSERS.parse_cupdlpc_json(output)
    assert float(row["rel_gap"]) == 2.9e-5


def test_cupdlpc_accepts_nonstandard_nonfinite_tokens(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    output.write_text(
        '{"terminationCode":"TIMELIMIT_OR_ITERLIMIT",'
        '"dSolvingTime":1000.0,"dRelPrimalFeas":1.0,'
        '"dRelDualFeas":inf,"dRelDualityGap":1.0}',
        encoding="utf-8",
    )
    row = PARSERS.parse_cupdlpc_json(output)
    assert row["status"] == "TIMELIMIT_OR_ITERLIMIT"
    assert row["rel_dual"] == "inf"


def test_hprlp_julia_expands_common_residual(tmp_path: Path) -> None:
    output = tmp_path / "one.csv"
    output.write_text(
        "solver,split,instance,status,time_sec,iterations,objective,rel_gap,rel_residual,error\n"
        "HPR-LP(Julia),Small,test,OPTIMAL,1.0,20,0.0,3e-6,4e-6,\n",
        encoding="utf-8",
    )
    row = PARSERS.parse_hprlp_jl_csv(output)
    assert float(row["rel_primal"]) == 4e-6
    assert float(row["rel_dual"]) == 4e-6


def test_cupdlp_julia_reads_convergence_candidate(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    output.write_text(
        json.dumps(
            {
                "termination_string": "OPTIMAL",
                "iteration_count": 448,
                "solve_time_sec": 1.8,
                "solution_stats": {
                    "convergence_information": [
                        {
                            "primal_objective": -120.0,
                            "relative_optimality_gap": 9.0e-5,
                            "relative_l2_primal_residual": 1.2e-5,
                            "relative_l2_dual_residual": 0.0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    row = PARSERS.parse_cupdlp_jl_json(output)
    assert float(row["rel_gap"]) == 9.0e-5
    assert float(row["rel_primal"]) == 1.2e-5
    assert float(row["rel_dual"]) == 0.0


def test_group_runner_isolates_manifest_stdin_for_every_solver() -> None:
    assert RUNNER_PATH.read_text(encoding="utf-8").count("</dev/null") == 5


def test_cupdlpx_resolves_dotted_instance_output_stem(tmp_path: Path) -> None:
    expected = tmp_path / "thk.48_summary.txt"
    actual = tmp_path / "thk_summary.txt"
    actual.write_text("Termination Reason: OPTIMAL\n", encoding="utf-8")
    assert PARSERS.resolve_solver_output("cupdlpx_c", expected) == actual


def test_cupdlpx_reads_native_primal_objective_value_label(tmp_path: Path) -> None:
    output = tmp_path / "summary.txt"
    output.write_text(
        "Termination Reason: OPTIMAL\n"
        "Runtime (sec): 0.2\n"
        "Iterations Count: 40\n"
        "Primal Objective Value: -2.829603e+02\n"
        "Relative Objective Gap: 1e-7\n"
        "Relative Primal Residual: 2e-7\n"
        "Relative Dual Residual: 3e-7\n",
        encoding="utf-8",
    )
    row = PARSERS.parse_cupdlpx_txt(output)
    assert float(row["objective"]) == -282.9603
