# FA_CP

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**FA_CP** is a GPU-accelerated fully augmented fixed-point/correction method
for large-scale convex quadratic programming (QP). It uses the same problem
interface and benchmark conventions as restarted primal-dual QP baselines,
while replacing the non-augmented primal step with an inexact solve of the full
augmented-Lagrangian primal subproblem. The AL penalty `sigma` is an explicit
solver parameter; `sigma = 0` is handled by a separate non-augmented branch
used as an internal ablation.

The FA_CP algorithmic path, guarded penalty policy, experiment wrappers, and
release interface in this directory are maintained as Benqi Liu's research
implementation. Required notices for retained low-level components are kept
separately from the solver's public method identity; see [NOTICE](NOTICE).

---

## Problem Formulation

FA_CP solves convex quadratic programs in the following form, which allows a flexible input of a sparse quadratic term and a low-rank quadratic term:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

## FA_CP Update

FA_CP uses an augmented-Lagrangian primal step for

```math
\min_{x \in X} \; \frac{1}{2} x^\top Q x + c^\top x
\quad \text{s.t.} \quad Ax \in S.
```

For a dual stepsize `rho_k`, primal stepsize `tau_k`, and explicit positive AL penalty parameter `sigma`, the primal step solves the full AL proximal subproblem

```math
x^{k+1} \approx \arg\min_{x \in X}
\left\{
\frac{1}{2} x^\top Q x + c^\top x
+ \frac{\sigma}{2}\mathrm{dist}^2\!\left(Ax-\frac{y^k}{\sigma}, S\right)
+ \frac{1}{2\tau_k}\|x-x^k\|^2
\right\},
```

where the inner solve uses the BB-style routine and recomputes the AL interaction at every inner iterate instead of freezing it at `x^k`. The reflected primal iterate is

```math
\bar x^{k+1} = 2x^{k+1} - x^k.
```

The dual update for positive `sigma` is the closed-form `FA_CP` proximal step

```math
y^{k+1}
= y^k + \rho_k\left(
\Pi_S\!\left(A\bar x^{k+1} - \frac{y^k}{\sigma + \rho_k}\right)
- A\bar x^{k+1}
\right).
```

The restart rule, primal-weight update, and fixed-point error remain compatible
with the corresponding non-augmented restarted primal-dual baseline.

When `sigma` is zero, the implementation uses the non-augmented branch rather
than evaluating the AL formulas with `sigma = 0`. This branch is intended as an
ablation path inside the same rescaled solver framework; it is not a claim of
bitwise or implementation-level equivalence to any external baseline codebase.

## Penalty semantics and experiment scope

The release uses `sigma` in two deliberately separate experiments.

### Controlled theoretical prediction

Under the identified affine-manifold and simultaneous-diagonalization
assumptions in the paper, the mode-wise critical value is

```math
\sigma_i^{\mathrm{crit}}
= \frac{\max\{2\sqrt{\rho s_i^2(p-\rho s_i^2)}-h_i,0\}}{s_i^2}.
```

The problem-level one-step prediction is not one selected mode. It minimizes
the worst modal spectral radius,

```math
\widehat\sigma_1
= \arg\min_{\sigma\in[0,2]}\max_i \rho(J_i(\sigma)).
```

The restarted-epoch experiment analogously computes `sigma_hat_K` for the
complete reflected-Halpern epoch polynomial. These predictions are reproduced
by the self-contained scripts in
[`../../experiments/modal_sigma`](../../experiments/modal_sigma).

### Practical 134-instance benchmark policy

The 134-instance Maros--Meszaros run does not estimate the modal spectra and does
not evaluate the formula above. It initializes `sigma = 0` and uses the
committed `guarded` restart-only rule, which activates a positive value only
when its residual and movement safeguards are satisfied. Restart-candidate
selection is disabled by default in the batch runner.

The standalone solver reads QP instances in MPS/QPS form and accepts row bounds
directly. This is the slack-eliminated realization of the paper's equality
model: for the row set \(S=[\ell_c,u_c]\), introduce \(z\in S\) and the
equality \(Ax-z=0\). Eliminating \(z\) from the augmented step gives the box
projection displayed above. The GPU implementation evaluates that projection
without materializing slack columns. The Maros--Meszaros scripts read the
prepared Table 1 files supplied through `DATA_DIR` through this path.


## Installation (C++ Executable)

To use the standalone C++ solver, you must compile the project using CMake.

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+.
* **Build Tools:** CMake (≥ 3.24), GCC, NVCC.

### Build from Source
Clone the repository and compile the QP solver using CMake.
```bash
git clone https://github.com/bqliu815/PDAL.git
cd PDAL/QP/FA_CP
cmake -S . -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/fa_cp` or `./build/bin/fa_cp`, depending on the generator and install layout.

If your system has multiple CUDA versions or the default nvcc is outdated (e.g., in `/usr/bin/nvcc`), you should explicitly specify the path to your modern CUDA compiler using the CUDACXX environment variable.
```bash
git clone https://github.com/bqliu815/PDAL.git
cd PDAL/QP/FA_CP
# Replace '/your/path/to/nvcc' with the actual path, e.g., /usr/local/cuda-12.6/bin/nvcc
CUDACXX=/your/path/to/nvcc cmake -S . -B build
cmake --build build --clean-first
```

##  Usage (C++ Executable)

Run the solver from the command line:

```bash
./build/bin/fa_cp <MPS_FILE> <OUTPUT_DIR> [OPTIONS]
```

### Command Line Arguments

**Positional Arguments:**

1. `<MPS_FILE>`: Path to the input QP (supports `.mps`, `.qps`, and `.mps.gz`).
2. `<OUTPUT_DIR>`: Directory where solution files will be saved.

Solver Parameters:
| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| -h, --help | flag | Display the help message. | N/A |
| -v, --verbose | int | Verbosity level: 0 (Silent), 1 (Summary), 2 (Detailed). | 1 |
| --time_limit | double | Time limit in seconds. | 3600.0 |
| --iter_limit | int | Iteration limit. | 2147483647 |
| --eps_opt | double | Relative optimality tolerance. | 1e-4 |
| --eps_feas | double | Relative feasibility tolerance. | 1e-4 |
| --eps_infeas_detect | double | Infeasibility detection tolerance. | 1e-10 |
| --l_inf_ruiz_iter | int | Iterations for L-inf Ruiz rescaling. | 10 |
| --pock_chambolle_alpha | double | Value for Pock-Chambolle step size parameter $\alpha$. | 1.0 |
| --no_pock_chambolle | flag | Disable Pock-Chambolle rescaling (enabled by default). | false |
| --no_bound_obj_rescaling | flag | Disable bound objective rescaling (enabled by default). | false |
| --no_al_qp | flag | Disable the augmented-Lagrangian update and use the non-augmented baseline update. | false |
| --al_sigma | float | Initial AL penalty parameter used in the full AL subproblem. | 0 |
| --sigma_update | string | Restart-only sigma update rule. | guarded |
| --sv_max_iter | int | Max iterations for singular value estimation (Power Method). | 5000 |
| --sv_tol | double | Tolerance for singular value estimation. | 1e-4 |
| --eval_freq | int | Frequency of termination criteria evaluation (in iterations). | 200 |
| --opt_norm | string | Norm for optimality criteria (l2 or linf). | linf |

---

## Maros--Meszaros batch runner

The runner executes one solver binary and one policy over a prepared list of
instances:

```bash
cd QP/FA_CP
sbatch --export=ALL,DATA_DIR=/path/to/Maros-Meszaros-Table1-134 \
  run_table1_134.sbatch
```

The default configuration is explicit in the script:

```text
EPS_OPT=1e-6
EPS_FEAS=1e-6
TIME_LIM=1000
EVAL_FREQ=200
FA_CP_USE_AUGMENTATION=1
FA_CP_SIGMA=0
FA_CP_SIGMA_UPDATE=guarded
FA_CP_RESTART_SELECT=0
```

Generated summaries and logs are written below the user-selected output root
and are not tracked by this repository.

---

## Python Interface

FA_CP provides a user-friendly Python interface that allows you to define, solve, and analyze QP problems using familiar libraries like NumPy and SciPy.

For detailed instructions on how to use the Python interface, including installation, modeling, and examples, please see the [Python Interface README](./python/README.md).

### Quick Example in Python

```python
import numpy as np
import scipy.sparse as sp
from fa_cp import Model

# Example: minimize 0.5 * x'(Q + R'R)x + c'x
# subject to l <= A x <= u,  lb <= x <= ub

# 1. Define Standard QP terms
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])

# 2. Define Low-Rank Matrix R
# Let's add a term 0.5 * ||Rx||^2 where R = [[1, 0]]
# This adds 0.5 * (x1)^2 to the objective
R = sp.csc_matrix([[1.0, 0.0]])

# 3. Define Constraints
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])

# 4. Create QP model with Low-Rank term (R), where Q and R are both optional.
m = Model(objective_matrix=Q,
          objective_matrix_low_rank=R,  
          objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)

# 5. Set solver parameters (0=Silent, 1=Summary, 2=Detailed)
m.setParams(LogLevel=2)

# Solve
m.optimize()

# Print results
print(f"Status: {m.Status}")
print(f"Objective: {m.ObjVal:.4f}")
if m.X is not None:
    print(f"Primal Solution: {m.X}")
```

## Citation

If you use this software or method in your research, please cite the
accompanying paper. Repository-wide citation metadata is available in
[`../../CITATION.cff`](../../CITATION.cff).

Methodological comparisons and baseline citations are given in the paper.
Software-origin notices required for redistribution are collected in
[NOTICE](NOTICE) and the repository-level
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).



---

## License

Copyright 2026 Benqi Liu.

Required notices for retained Apache-2.0 low-level components are provided in
[NOTICE](NOTICE), [LICENSE](LICENSE), and the corresponding source headers.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
