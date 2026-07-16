from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_fa_cp_inner_solver_matches_the_experimental_disclosure() -> None:
    state = source("QP/FA_CP/src/solver_state.cu")
    core = source("QP/FA_CP/src/fa_cp_core_op.cu")
    inner_step = between(
        core,
        "void primal_BB_step_size_update(",
        "void primal_gradient_update(",
    )

    assert "iteration_limit = 1000" in state
    assert "tol = 1e-3" in state
    assert (
        "while (inner_solver_iter < state->inner_solver->iteration_limit)"
        in inner_step
    )
    assert inner_step.count("compute_al_effective_dual_product(") >= 2
    assert "fmin(" in core
    assert "0.0005 * primal_norm" in core
    assert "1e-9" in core


def test_linearized_qp_maps_use_one_explicit_primal_gradient_step() -> None:
    core = source("QP/LIN_CP_AL/src/cp_al_core_op.cu")
    update = between(core, "void cp_al_update(", "void halpern_update(")

    assert "primal_gradient_update(state, primal_step_size);" in update
    assert "primal_BB_step_size_update(" not in update
    assert "if (state->use_al_qp)" in update


def test_planted_subproblem_rows_match_the_inner_solve_disclosure() -> None:
    for directory in (
        "QP/PLANTED_CP_AL",
        "QP/PLANTED_FA_CP",
    ):
        state = source(f"{directory}/src/solver_state.cu")
        core_name = "cp_al_core_op.cu" if "CP_AL" in directory else "fa_cp_core_op.cu"
        core = source(f"{directory}/src/{core_name}")
        assert "iteration_limit = 1000" in state
        assert "tol = 1e-3" in state
        assert "fmin(" in core
        assert "0.0005 * primal_norm" in core

    cpal_core = source("QP/PLANTED_CP_AL/src/cp_al_core_op.cu")
    cpal_inner = between(
        cpal_core,
        "void primal_BB_step_size_update(",
        "void primal_gradient_update(",
    )
    cpal_update = between(cpal_core, "void cp_al_update(", "void halpern_update(")
    assert "compute_al_effective_dual_product(" not in cpal_inner
    assert cpal_update.count("compute_al_effective_dual_product(") == 1

    facp_core = source("QP/PLANTED_FA_CP/src/fa_cp_core_op.cu")
    facp_inner = between(
        facp_core,
        "void primal_BB_step_size_update(",
        "void primal_gradient_update(",
    )
    assert facp_inner.count("compute_al_effective_dual_product(") >= 2


def test_planted_table_uses_one_native_timer_implementation() -> None:
    for directory in (
        "QP/PLANTED_CP_AL",
        "QP/PLANTED_FA_CP",
        "QP/LIN_CP_AL",
    ):
        solver = source(f"{directory}/src/solver.cu")
        assert "clock_t start_time = clock();" in solver
        assert "(double)(clock() - start_time) / CLOCKS_PER_SEC" in solver


def test_public_qp_profiles_cover_all_five_table_rows() -> None:
    profiles = source("experiments/qp_linearized/profiles.psv").splitlines()
    assert profiles[0] == (
        "name|source_dir|binary_name|binary_env|eval_freq|extra_args"
    )
    assert len(profiles[1:]) == 5
    assert {line.split("|")[1] for line in profiles[1:]} == {
        "",
        "QP/PLANTED_CP_AL",
        "QP/PLANTED_FA_CP",
        "QP/LIN_CP_AL",
    }
    assert profiles[1].split("|")[3] == "PDHCG_II_BIN"
    assert "--no_al_qp" in profiles[4]
    runner = source("experiments/qp_linearized/run_qp_linearized.sbatch")
    assert "profile_count=" in runner
    assert "profile_index > profile_count" in runner
    assert "--array=0-199" in runner


def test_research_solver_banners_use_the_released_method_names() -> None:
    expected = {
        "QP/PLANTED_CP_AL/src/utils.cu": "RHR-CP-AL",
        "QP/PLANTED_FA_CP/src/utils.cu": "RHR-FA-CP",
        "QP/LIN_CP_AL/src/utils.cu": "RHR-Lin-CP-AL",
    }
    for path, method in expected.items():
        contents = source(path)
        assert method in contents
        assert "(c) Benqi Liu, 2026" in contents
        assert "ishongpeili@gmail.com" not in contents
