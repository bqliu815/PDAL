# RHR-FA-CP planted-KKT implementation

This directory contains Benqi Liu's subproblem-based FA_CP implementation for
equality-constrained box QPs. It is separate from `QP/FA_CP`, which includes a
guarded restart-only penalty option.

Build from the repository root with:

```bash
cmake -S QP/PLANTED_FA_CP -B QP/PLANTED_FA_CP/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build QP/PLANTED_FA_CP/build -j
```

The command-line executable is `QP/PLANTED_FA_CP/build/rhr_fa_cp`.

Required notices for retained low-level components are kept in
[NOTICE](NOTICE), [LICENSE](LICENSE), and the modified source-file headers.
