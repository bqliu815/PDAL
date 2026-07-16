from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "experiments" / "random_lp" / "build_random_lp_suite.py"
MANIFEST_AUDITOR = ROOT / "experiments" / "random_lp" / "make_suite_manifest.py"
RUNNER = ROOT / "experiments" / "random_lp" / "run_random_lp.sbatch"
CPAL_CORE = ROOT / "LP" / "CP_AL" / "src" / "solver.cu"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_small_suite(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--out_dir",
            str(path),
            "--per_family",
            "1",
            "--small_weight",
            "1",
            "--medium_weight",
            "0",
            "--large_weight",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_fixed_suite_is_deterministic_and_portable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_small_suite(first)
    build_small_suite(second)

    first_manifest = first / "suite_manifest.tsv"
    second_manifest = second / "suite_manifest.tsv"
    assert first_manifest.read_bytes() == second_manifest.read_bytes()

    with first_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["family"] for row in rows] == ["easy", "scaled", "neardep", "combo"]
    assert [int(row["seed"]) for row in rows] == [2026042401, 2026042402, 2026042403, 2026042404]
    assert all(row["size"] == "s" for row in rows)
    assert all(not Path(row["mps"]).is_absolute() for row in rows)

    for row in rows:
        assert file_sha256(first / row["mps"]) == file_sha256(second / row["mps"])

    input_hashes = first / "input_hashes.tsv"
    subprocess.run(
        [
            sys.executable,
            str(MANIFEST_AUDITOR),
            "--suite-dir",
            str(first),
            "--per-family",
            "1",
            "--hash-output",
            str(input_hashes),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with input_hashes.open(newline="", encoding="utf-8") as handle:
        hash_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(hash_rows) == 4
    assert all(
        row["sha256"] == file_sha256(first / row["mps"])
        for row in hash_rows
    )


def test_group_runner_isolates_manifest_stdin(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    for instance in ("first", "second"):
        (suite / f"{instance}.mps").write_text("NAME TEST\nENDATA\n", encoding="utf-8")
    manifest = tmp_path / "suite.tsv"
    manifest.write_text(
        "family\tsize\tinstance\tseed\tm\tn\tnnz\tdensity_realized\tmps\n"
        "easy\ts\tfirst\t1\t1\t1\t1\t1\tfirst.mps\n"
        "easy\ts\tsecond\t2\t1\t1\t1\t1\tsecond.mps\n",
        encoding="utf-8",
    )
    fake_solver = tmp_path / "fake_solver.sh"
    fake_solver.write_text(
        "#!/bin/bash\n"
        "read -r _ || true\n"
        "previous=\n"
        "for argument in \"$@\"; do second_last=$previous; previous=$argument; done\n"
        "mps=$second_last\n"
        "out=$previous\n"
        "name=$(basename \"$mps\" .mps)\n"
        "mkdir -p \"$out\"\n"
        "printf 'Termination Reason: OPTIMAL\\n' > \"$out/${name}_summary.txt\"\n",
        encoding="utf-8",
    )
    fake_srun = tmp_path / "fake_srun.sh"
    fake_srun.write_text(
        "#!/bin/bash\nshift 2\nexec \"$@\"\n",
        encoding="utf-8",
    )
    fake_module = tmp_path / "module"
    fake_module.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_solver.chmod(0o755)
    fake_srun.chmod(0o755)
    fake_module.chmod(0o755)
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "MANIFEST": str(manifest),
            "SUITE_DIR": str(suite),
            "CPAL_BIN": str(fake_solver),
            "CUPDLPX_BIN": str(fake_solver),
            "OUT_ROOT": str(output),
            "GROUP_SIZE": "2",
            "SLURM_ARRAY_TASK_ID": "0",
            "SRUN": str(fake_srun),
            "PATH": f"{tmp_path}:{env['PATH']}",
        }
    )
    subprocess.run(["bash", str(RUNNER)], check=True, env=env)
    assert sorted(path.name for path in (output / "runs").iterdir()) == [
        "first",
        "second",
    ]


def test_random_lp_sigma_rule_matches_the_documented_protocol() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    core = CPAL_CORE.read_text(encoding="utf-8")

    for setting in (
        "CP_AL_SIGMA_BALANCE_GAIN=0",
        "CP_AL_SIGMA_KP=0.5",
        "CP_AL_SIGMA_KI=0",
        "CP_AL_SIGMA_KD=0",
    ):
        assert setting in runner
    assert "0.5 * log(fmax(dx_metric_sq" in core
    assert "sigma_pid_kp * sigma_error" in core
    assert "state->al_sigma = state->al_sigma_best;" in core
