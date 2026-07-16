from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_completed_run_artifacts_are_not_committed() -> None:
    assert not (ROOT / "results").exists()
    assert not (ROOT / "FROZEN_HASHES.md").exists()
    assert not (ROOT / "PAPER_REPRODUCIBILITY.md").exists()

    forbidden_names = {
        "audit.json",
        "source_provenance.json",
        "reference_metadata.json",
        "reference_manifest.csv",
    }
    forbidden_suffixes = (
        "_runs.csv",
        "_aggregate.csv",
        "_table.csv",
        "_table_rows.tex",
    )
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        assert path.name not in forbidden_names, path.relative_to(ROOT)
        assert not path.name.endswith(forbidden_suffixes), path.relative_to(ROOT)


def test_generated_output_locations_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (
        "results/",
        "paper_runs/",
        "experiments/qp_linearized/reference_manifest.csv",
        "experiments/qp_linearized/reference_metadata.json",
        "experiments/qp_linearized/source_provenance.json",
    ):
        assert entry in ignore
