# LP benchmark runners

This directory provides wrappers and parsers for running CP_AL and external LP
solvers on a common task manifest. Benchmark matrices, third-party source
trees, executables, and completed outputs are not included.

## Prepare a manifest

Generate the LP task manifest with
`LP/CP_AL/make_full856_manifest.py`, using local MIPLIB and Mittelmann roots.
Use the same manifest for CP_AL and every comparison solver.

## Run comparison solvers

Install the desired third-party solvers separately, then set their executable
or project paths as documented in `run_baseline_group.sbatch`:

```bash
BASELINE_ROOT=/path/to/third_party_install \
JULIA=/path/to/julia \
MANIFEST=/path/to/full856.tsv \
RUN_ROOT=/path/to/baseline_run \
bash experiments/lp_benchmarks/submit_lp_baselines.sh
```

The CP_AL runner is documented in [`LP/CP_AL`](../../LP/CP_AL).

## Collect outputs

```bash
python experiments/lp_benchmarks/collect_lp_tables.py \
  --manifest /path/to/full856.tsv \
  --baseline-run-root /path/to/baseline_run \
  --cpal-run-root /path/to/cpal_run \
  --output-dir /path/to/lp_aggregate
```

The collector parses native solver summaries, validates the common task set,
and writes normalized rows and aggregate statistics to the selected local
output directory.
