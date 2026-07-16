# CP_AL and FA_CP

[![Code checks](https://github.com/bqliu815/PDAL/actions/workflows/code-checks.yml/badge.svg)](https://github.com/bqliu815/PDAL/actions/workflows/code-checks.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository provides the source code accompanying "Restarted Reflected
Halpern Acceleration for Augmented Primal--Dual Methods." It contains Benqi
Liu's implementations of the **CP_AL** and **FA_CP** method families, including
subproblem-based and linearized variants.

Generated instances, solver outputs, aggregate tables, benchmark logs, and
third-party executables are not included in this repository.

## Source layout

- [`LP/CP_AL`](LP/CP_AL): CP_AL solver and LP runner.
- [`QP/FA_CP`](QP/FA_CP): FA_CP solver and Python interface.
- [`QP/LIN_CP_AL`](QP/LIN_CP_AL): linearized CP_AL solver and non-augmented
  control.
- [`QP/PLANTED_CP_AL`](QP/PLANTED_CP_AL): subproblem-based CP_AL QP solver.
- [`QP/PLANTED_FA_CP`](QP/PLANTED_FA_CP): subproblem-based FA_CP QP solver.
- [`experiments`](experiments): input generators, batch runners, and output
  collectors.

## Build and run

Clone the repository and follow the README in the relevant solver directory:

```bash
git clone https://github.com/bqliu815/PDAL.git
cd PDAL
```

The experiment guides under [`experiments`](experiments) describe the expected
command-line inputs. All generated files are written to user-selected local
directories and are ignored by Git.

## Tests

The code-level test suite does not require committed experiment outputs:

```bash
python -m pip install pytest -r experiments/random_lp/requirements.txt
python -m pytest -q
```

CUDA-dependent tests in the solver subdirectories require the corresponding
extensions to be built first.

## Citation

Please cite the accompanying paper when using this software. Citation metadata
is provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the Apache License 2.0; see
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
