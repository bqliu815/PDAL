# Random equality-box LP experiment

This directory provides deterministic generators, a paired GPU runner, and a
collector for four synthetic equality-constrained box-LP families. Generated
MPS files and solver outputs are not included.

## Generate inputs

```bash
python -m pip install -r experiments/random_lp/requirements.txt
python experiments/random_lp/build_random_lp_suite.py \
  --out_dir /path/to/random_lp_suite
```

For a small smoke test, add
`--per_family 1 --small_weight 1 --medium_weight 0 --large_weight 0`.

To rebuild a portable manifest for an existing generated suite:

```bash
python experiments/random_lp/make_suite_manifest.py \
  --suite-dir /path/to/random_lp_suite \
  --hash-output /path/to/random_lp_suite/input_hashes.tsv
```

## Run the solvers

Provide locally built CP_AL and comparison executables:

```bash
rows=$(( $(wc -l < /path/to/random_lp_suite/suite_manifest.tsv) - 1 ))
group_size=10
groups=$(( (rows + group_size - 1) / group_size ))

sbatch --array=0-$((groups - 1)) \
  --export=ALL,MANIFEST=/path/to/random_lp_suite/suite_manifest.tsv,\
SUITE_DIR=/path/to/random_lp_suite,CPAL_BIN=/path/to/cp_al,\
CUPDLPX_BIN=/path/to/comparison_solver,OUT_ROOT=/path/to/run,\
GROUP_SIZE=$group_size \
  experiments/random_lp/run_random_lp.sbatch
```

## Collect outputs

```bash
python experiments/random_lp/collect_random_lp.py \
  --manifest /path/to/random_lp_suite/suite_manifest.tsv \
  --run-root /path/to/run \
  --output-dir /path/to/aggregate
```

The collector validates the run layout and writes normalized rows, aggregate
statistics, and a local audit file to the selected output directory.
