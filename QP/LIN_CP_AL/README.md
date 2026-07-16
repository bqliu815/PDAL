# RHR-Lin-CP-AL solver

This directory contains Benqi Liu's GPU implementation of linearized CP_AL for
equality-constrained box QPs. Its non-augmented `--no_al_qp` branch provides the
corresponding non-augmented control without duplicating the numerical core.

## Method represented by this source

The solver accepts convex QPs of the form

```text
minimize    0.5 x' H x + c' x
subject to  A x = b,  lower <= x <= upper.
```

For a QP objective, the primal update evaluates `H x` at the current iterate
and takes one projected gradient/proximal step. The augmented interaction is
also evaluated explicitly through the effective multiplier. Thus this source
implements the paper's **linearized CP-AL/PFBS map**; it does not retain the
quadratic objective or augmented term in an inner primal subproblem. The
subproblem-based `RHR-CP-AL` and `RHR-FA-CP` rows use different code paths.

## Build

The build requires CMake 3.24 or newer, a C/C++ compiler, CUDA, cuBLAS,
cuSPARSE, and zlib.

```bash
cmake -S QP/LIN_CP_AL -B QP/LIN_CP_AL/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build QP/LIN_CP_AL/build -j
```

The executable is `QP/LIN_CP_AL/build/rhr_lin_cp_al`.

## Paper profile

The paper row fixes the same parameters for all 1000 instances:

```text
--time_limit 60
--eps_opt 1e-5
--eps_feas 1e-5
--sigma_init 0.1
--reflection_coefficient 0.7
--pock_chambolle_alpha 1.5
--eval_freq 400
```

Here `--sigma_init 0.1` is the requested initial penalty. After scaling, the
solver projects this value onto its instance-wise spectral safety interval
`[sigma_base, 100 sigma_base]` and keeps the resulting penalty fixed. The same
rule is applied to every instance.

Use
[`experiments/qp_linearized/run_qp_linearized.sbatch`](../../experiments/qp_linearized/run_qp_linearized.sbatch)
to run the common five-method planted-KKT protocol. The lower-level
`run_table1_134.sbatch` remains available as a per-list diagnostic wrapper.

## License and attribution

This source is distributed under the Apache License 2.0. The RHR-Lin-CP-AL
algorithmic modifications and release interface are Copyright 2026 Benqi Liu.
Required notices for retained low-level components are provided in
[LICENSE](LICENSE), [NOTICE](NOTICE), and the modified source-file headers.
