# RHR-CP-AL subproblem implementation

This directory contains Benqi Liu's subproblem-based CP_AL implementation for
equality-constrained box QPs. It is distinct from the explicit
`RHR-Lin-CP-AL` implementation in `QP/LIN_CP_AL`.

Build from the repository root with:

```bash
cmake -S QP/PLANTED_CP_AL -B QP/PLANTED_CP_AL/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build QP/PLANTED_CP_AL/build -j
```

The command-line executable is `QP/PLANTED_CP_AL/build/rhr_cp_al`.

Required notices for retained low-level components are kept in
[NOTICE](NOTICE), [LICENSE](LICENSE), and the modified source-file headers.
