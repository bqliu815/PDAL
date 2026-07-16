# CP_AL

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

`CP_AL` is a GPU implementation of a restarted reflected-Halpern
Chambolle--Pock augmented-Lagrangian method for linear programs of the form

```math
\min_x c^\top x \quad \text{s.t.} \quad Ax \in [l,u],
\quad x \in [\ell_x,u_x].
```

The solver uses a direct `Ax` box-projection path and supports both the
non-augmented lifted-PDHG corner and positive CP_AL augmentation.

## Augmentation parameterization

The main implementation parameterizes augmentation as a fraction of the dual
proximal coefficient. For dual step `rho_k`, the code sets

```math
\beta_k = \rho_k\frac{\alpha}{1-\alpha},
```

where `beta_k` is the augmented-Lagrangian penalty. The historical environment
variable name is

```bash
CP_AL_LAMBDA=0.05
```

but its value is the fraction `alpha`, not the metric coefficient denoted by
`lambda` below. Thus `CP_AL_LAMBDA=0` gives the lifted-PDHG update, while any
positive value gives a positive augmented penalty.

The direct-box predictor is

```math
u^k = y^k - \beta_k\left(Ax^k-
\Pi_{[l,u]}\left(Ax^k-\frac{y^k}{\beta_k}\right)\right),
```

and the dual update is

```math
y^{k+1}=y^k-\rho_k\left(A\bar x^{k+1}-
\Pi_{[l,u]}\left(A\bar x^{k+1}-
\frac{y^k}{\rho_k+\beta_k}\right)\right).
```

For controlled experiments using the paper's scalar metric parameterization,
set `CP_AL_SIGMA_MODE=1` or provide `CP_AL_SIGMA`. In that mode,

```math
\rho_k=\sigma_k, \qquad
\tau_k=\frac{1}{\sigma_k\lambda}, \qquad
\lambda\ge 2\|A\|^2.
```

The default is `lambda = 2.01 ||A||^2`; it can be changed with
`CP_AL_LAMBDA_PARAM` or `CP_AL_LAMBDA_FACTOR`. This sigma mode is available for
experiments but is not the path used by the released LP benchmark controller.

## Build

Requirements are an NVIDIA GPU, CUDA 12.4 or newer, CMake 3.24 or newer, and a
compatible C/C++ compiler.

```bash
git clone https://github.com/bqliu815/PDAL.git
cd PDAL/LP/CP_AL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first -j
```

The command-line executable is written to `build/cp_al`.

## Released LP benchmark controller

The paper benchmark entry point is [`run_unified_full856.sbatch`](run_unified_full856.sbatch).
It enables `CP_AL_UNIFIED_POLICY=1`, clears inherited `CP_AL_*` settings, and
runs one executable on all 856 instance-tolerance tasks. The controller uses
only numerical features of the parsed LP and uniform iteration state. It does
not use the instance name, file path, benchmark, split, requested tolerance,
manifest position, node, or elapsed wall time.

The frozen decision list is:

| Route | Anonymous condition | Action |
| --- | --- | --- |
| `lowobj_highcv_eval100_cr5` | `objective_density <= 0.01` and `col_nnz_cv >= 5` | evaluation frequency 100 and Curtis--Reid effort 5 |
| `zero_equality_cr0` | otherwise, `equality_fraction == 0` | evaluation frequency 200 and Curtis--Reid effort 0 |
| `high_equality_tail_or_base` | all remaining matrices | run the base policy; activate the structural tail only if its internal predicate holds |

The structural tail predicate is

```text
objective_density >= 0.99
and row_logmax_std <= 0.05
and (equality_fraction <= 0.001 or equality_fraction >= 0.99).
```

Only this tail can activate positive augmentation. Its continuation contains
one deterministic block with `alpha = 0.002` for 200 charged iterations after
the fixed iteration/epoch gate. Nonmatching matrices use the base lifted-PDHG
solve with `alpha = 0`. Reflection and restart thresholds are fixed globally;
only the route actions listed above vary with anonymous structure.

Every benchmark invocation passes `--no_presolve`. The MIPLIB inputs are the
already prepared Gurobi-presolved LP relaxations; CP_AL itself performs no
additional presolve stage in this protocol.

The source guards for the policy boundary are run with

```bash
python3 -m pytest -q \
  test/test_feature_policy_source.py \
  test/test_manifest_builder.py \
  test/test_structural_tail_cut_source.py
```

These rules and constants are the complete controller used for the released
aggregate.

## Reproduce the 856-task run

Arrange the inputs as

```text
/path/to/MIPLIB2017_gurobifinish/
  Small/       # 268 MPS files
  Medium/      # 93 MPS files
  Large/       # 18 MPS files

/path/to/mittelmann/  # 49 MPS files
```

Build the deterministic task manifest:

```bash
python3 make_full856_manifest.py \
  --miplib-root /path/to/MIPLIB2017_gurobifinish \
  --mittelmann-root /path/to/mittelmann \
  --output full856.tsv
```

Then submit the single-policy array:

```bash
sbatch --array=0-855 \
  --export=ALL,MANIFEST=$PWD/full856.tsv,BIN=$PWD/build/cp_al,OUT_ROOT=$PWD/run \
  run_unified_full856.sbatch
```

Override the Slurm partition and resource flags as needed for the target
system. Set `CUDA_MODULE`, `CUDA_HOME`, or `SRUN` when they are not discoverable
from the environment.

The protocol uses tolerances `1e-4` and `1e-8`; time limits are 3600 seconds
for MIPLIB Small/Medium, 18000 seconds for MIPLIB Large, and 1000 seconds for
Mittelmann. Public data sets and third-party baseline executables are not
redistributed.

All generated manifests and solver outputs remain in the user-selected local
directories and are not tracked by this repository.

## License and attribution

Copyright 2026 Benqi Liu. Licensed under the Apache License, Version 2.0.
Low-level GPU infrastructure retains its original Apache-2.0 notices; see
[NOTICE](NOTICE) and [LICENSE](LICENSE).
