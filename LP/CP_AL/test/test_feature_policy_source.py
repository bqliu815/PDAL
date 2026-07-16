# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.
"""Source-level boundary checks for the released numerical policy."""

from pathlib import Path


def marked_source(begin_marker: str, end_marker: str) -> str:
    source = (Path(__file__).resolve().parents[1] / "src" / "cli.c").read_text()
    start = source.index(begin_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_feature_policy_does_not_read_identity() -> None:
    body = marked_source(
        "// BEGIN_IDENTITY_FREE_FEATURE_POLICY",
        "// END_IDENTITY_FREE_FEATURE_POLICY",
    )

    forbidden = (
        "instance_name",
        "filename",
        "path_has_component",
        "str_eq(",
        "strstr(",
        "MIPLIB",
        "Mittelmann",
        '"Small"',
        '"Medium"',
        '"Large"',
    )
    for token in forbidden:
        assert token not in body, f"feature policy reads identity token: {token}"


def test_unified_policy_has_frozen_anonymous_routes() -> None:
    body = marked_source("// BEGIN_UNIFIED_POLICY", "// END_UNIFIED_POLICY")
    assert "stats.objective_density <= 0.01 && stats.col_nnz_cv >= 5.0" in body
    assert "stats.equality_fraction == 0.0" in body
    assert 'final_policy_setenv("CP_AL_UNIFIED_EVAL_FREQ", "100")' in body
    assert 'final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "5")' in body
    assert 'final_policy_setenv("CP_AL_FEATURE_CR_ITERS", "0")' in body
    assert 'final_policy_setenv("CP_AL_STRUCTURAL_TAIL_CUT", "1")' in body
    assert 'final_policy_setenv("CP_AL_TAIL_ONLY_PULSE", "1")' in body


def test_unified_policy_does_not_read_run_identity() -> None:
    body = marked_source("// BEGIN_UNIFIED_POLICY", "// END_UNIFIED_POLICY")
    code = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("//")
    )
    forbidden = (
        "instance_name",
        "filename",
        "dataset",
        "split",
        "tolerance",
        "manifest",
        "time_sec_limit",
        "getenv(",
        "strstr(",
        "strcmp(",
    )
    for token in forbidden:
        assert token not in code, f"unified policy reads run identity: {token}"


def test_full_runner_enables_only_the_unified_policy() -> None:
    runner = (
        Path(__file__).resolve().parents[1] / "run_unified_full856.sbatch"
    ).read_text()
    assert 'if [[ "$name" == CP_AL_* ]]' in runner
    assert "export CP_AL_UNIFIED_POLICY=1" in runner
    assert "--no_presolve" in runner
    assert "CP_AL_FINAL_AUTO_POLICY" not in runner
    assert "CP_AL_FEATURE_AUTO_POLICY" not in runner
    assert "case \"$instance\"" not in runner
    assert "if [[ \"$instance\"" not in runner
