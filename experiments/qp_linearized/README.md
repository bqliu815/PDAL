# Planted-KKT QP experiment

This directory contains the synthetic QP generator, five fixed profiles, a
batch runner, and an output collector. Generated QPS files and completed run
outputs are not included.

## Generate inputs

```bash
python -m pip install -r experiments/qp_linearized/requirements.txt
python experiments/qp_linearized/generate_synthetic_eqbox_qp_unified.py \
  --out-dir /path/to/qp_suite \
  --regime linearized_large \
  --total-instances 1000 \
  --seed 20260506
```

The generator plants a KKT point and writes the QPS files together with a
local manifest.

## Build and run

Build the three included solver trees:

```bash
for source in \
  QP/PLANTED_CP_AL QP/PLANTED_FA_CP QP/LIN_CP_AL; do
  cmake -S "$source" -B "$source/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$source/build" -j
done
```

The external comparison executable is not distributed. Supply it explicitly
when submitting the five-profile runner:

```bash
PDHCG_II_BIN=/path/to/comparison_solver \
DATASET_ROOT=/path/to/qp_suite \
OUT_ROOT=/path/to/qp_run \
sbatch experiments/qp_linearized/run_qp_linearized.sbatch
```

The fixed command-line profiles are listed in [`profiles.psv`](profiles.psv).

## Collect outputs

```bash
python experiments/qp_linearized/collect_qp_linearized.py \
  --run-root /path/to/qp_run/results \
  --manifest /path/to/qp_suite/manifest.csv \
  --output-dir /path/to/qp_aggregate
```

The collector validates method-instance coverage and writes normalized rows,
aggregate statistics, and a local audit file to the selected output directory.
