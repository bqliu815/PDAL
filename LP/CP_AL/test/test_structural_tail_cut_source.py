# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Source-level guards for the frozen v37 extreme-tail-only rule."""

from pathlib import Path


def controller_source() -> str:
    source = (Path(__file__).resolve().parents[1] / "src" / "cli.c").read_text()
    begin = source.index("// BEGIN_IDENTITY_FREE_STRUCTURAL_TAIL_CUT")
    end = source.index("// END_IDENTITY_FREE_STRUCTURAL_TAIL_CUT", begin)
    return source[begin:end]


def test_structural_tail_cut_has_exact_frozen_predicate() -> None:
    source = controller_source()
    assert "matrix_stats.equality_fraction <= 0.001" in source
    assert "matrix_stats.equality_fraction >= 0.99" in source
    assert "const bool homogeneous_dense_structure" in source
    assert "const bool structural_low_equality" in source
    assert "const bool structural_high_equality" in source
    assert "homogeneous_dense_structure && matrix_stats.equality_fraction <= 0.001" in source
    assert "homogeneous_dense_structure && matrix_stats.equality_fraction >= 0.99" in source
    assert "matrix_stats.objective_density >= 0.99" in source
    assert "matrix_stats.row_logmax_std <= 0.05" in source
    assert "const double base_fraction = structural_high_equality ? 0.05 : 0.75;" in source


def test_structural_tail_cut_has_frozen_profile_split() -> None:
    source = controller_source()
    assert 'const char *rescue_profile = "cr0";' in source
    assert "if (!structural_early_cut)" in source
    assert '"base_fraction=1.00 rescue_fraction=0.00 rescue_profile=none "' in source
    assert "cp_al_result_t *single_result = solve_lp_problem(problem, requested_params);" in source
    assert '"structural_early_cut=0 base_fraction=1.00 rescue_profile=none "' in source
    assert source.count('final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "0")') == 1
    assert "if (structural_early_cut)" in source
    assert '"[structural-tail-start] profile=%s remaining_limit=%.12e "' in source
    assert "remaining_limit = fmax(original_limit - base_solve_time, 0.0)" in source
    assert "rescue_params.termination_evaluation_frequency = 200" in source
    assert "rescue_params.termination_criteria.time_sec_limit = remaining_limit" in source


def test_positive_alpha_is_wired_only_into_the_tail_phase() -> None:
    full_source = (Path(__file__).resolve().parents[1] / "src" / "cli.c").read_text()
    source = controller_source()
    assert "bool tail_pulse_enabled" in source
    assert source.count('final_policy_setenv("CP_AL_ALPHA_DIRECT_PULSE", "0")') == 2
    assert source.count('final_policy_setenv("CP_AL_DETERMINISTIC_PULSE_FORCE", "0")') == 2
    assert "effective_tail_pulse_enabled =" in source
    assert "tail_pulse_enabled && structural_early_cut" in source
    assert source.count(
        '"CP_AL_ALPHA_DIRECT_PULSE", effective_tail_pulse_enabled ? "1" : "0"'
    ) == 1
    assert source.count(
        '"CP_AL_DETERMINISTIC_PULSE_FORCE", effective_tail_pulse_enabled ? "1" : "0"'
    ) == 1
    base_solve = source.index("solve_lp_problem(problem, &base_params)")
    rescue_solve = source.index("solve_lp_problem(problem, &rescue_params)")
    base_alpha_off = source.rindex(
        'final_policy_setenv("CP_AL_ALPHA_DIRECT_PULSE", "0")', 0, base_solve
    )
    assert base_alpha_off < base_solve
    assert base_solve < source.index(
        '"CP_AL_ALPHA_DIRECT_PULSE", effective_tail_pulse_enabled ? "1" : "0"'
    ) < rescue_solve
    assert full_source.count('env_flag_enabled("CP_AL_TAIL_ONLY_PULSE")') == 1
    assert "problem, &params, &feature_stats, tail_pulse_enabled" in full_source


def test_middle_profile_cannot_activate_positive_alpha() -> None:
    source = controller_source()
    early_definition = source.index(
        "const bool structural_early_cut =\n"
        "        structural_low_equality || structural_high_equality;"
    )
    effective_definition = source.index(
        "const bool effective_tail_pulse_enabled =\n"
        "        tail_pulse_enabled && structural_early_cut;"
    )
    base_fraction = source.index("const double base_fraction")
    assert early_definition < effective_definition < base_fraction
    assert "tail_pulse_requested=%d tail_pulse_enabled=%d" in source
    middle_begin = source.index("if (!structural_early_cut)")
    middle_end = source.index("cp_al_parameters_t base_params", middle_begin)
    middle = source[middle_begin:middle_end]
    assert 'final_policy_setenv("CP_AL_ALPHA_DIRECT_PULSE", "0")' in middle
    assert 'final_policy_setenv("CP_AL_DETERMINISTIC_PULSE_FORCE", "0")' in middle
    assert "solve_lp_problem(problem, requested_params)" in middle
    assert "phases=1 tail_started=0" in middle
    assert "rescue_profile=none" in middle
    assert "structural_tail_set_warm_anchor" not in middle
    assert "rescue_params" not in middle
    assert "remaining_limit" not in middle


def test_tail_pulse_eligibility_is_iteration_only() -> None:
    solver = (Path(__file__).resolve().parents[1] / "src" / "solver.cu").read_text()
    begin = solver.rindex("static bool alpha_direct_pulse_is_eligible(")
    end = solver.index("static void alpha_direct_pulse_start(", begin)
    eligible = solver[begin:end]
    for token in (
        "history->history_count < 2",
        "charged_count < 10000",
        "eval_frequency != 200",
        "remaining_iterations < 400",
        "const bool gate_passed = controller->force_pulse",
    ):
        assert token in eligible
    for token in (
        "cumulative_time_sec",
        "time_sec_limit",
        ".seconds",
        "remaining_seconds",
        "estimated_pulse_seconds",
        "previous_epoch_sec",
    ):
        assert token not in eligible


def test_structural_tail_cut_has_no_identity_or_tolerance_input() -> None:
    source = controller_source().lower()
    forbidden = (
        "instance_name",
        "filename",
        "dataset",
        "split",
        "tolerance",
        "miplib",
        "mittelmann",
        "manifest",
        "node",
        "gpu",
    )
    for token in forbidden:
        assert token not in source


def test_structural_tail_cut_has_no_runtime_rule_override() -> None:
    source = controller_source()
    forbidden = (
        "getenv(",
        "env_flag_enabled(",
        "BASE_FRACTION",
        "EARLY_CUT_THRESHOLD",
        "INSTANCE",
    )
    for token in forbidden:
        assert token not in source


def test_feature_statistics_are_computed_once_and_reused() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "cli.c").read_text()
    assert source.count("compute_feature_matrix_stats(problem)") == 1
    assert "feature_stats = compute_feature_matrix_stats(problem);" in source
    assert "apply_unified_policy(feature_stats_needed ? &feature_stats : NULL)" in source
    assert "apply_feature_auto_policy(problem, &params, feature_stats_needed ? &feature_stats : NULL)" in source
    assert "problem, &params, &feature_stats, tail_pulse_enabled" in source
    assert "apply_feature_auto_policy(problem, &rescue_params, precomputed_stats)" in source
    assert "[feature-matrix-stats] computed=1" in source


def test_structural_tail_uses_immutable_precomputed_stats() -> None:
    source = controller_source()
    assert "const feature_matrix_stats_t *precomputed_stats" in source
    assert "const feature_matrix_stats_t matrix_stats = *precomputed_stats;" in source
    assert "compute_feature_matrix_stats" not in source
