# Experiment entry points

This directory contains input generators, batch runners, parsers, and output
collectors. It does not contain generated instances or completed run outputs.

- [`modal_sigma`](modal_sigma): controlled one-step and restarted-epoch
  penalty experiments.
- [`random_lp`](random_lp): random equality-form box-LP generation and paired
  execution.
- [`lp_benchmarks`](lp_benchmarks): LP baseline wrappers and common-format
  aggregation.
- [`qp_linearized`](qp_linearized): planted-KKT QP generation and five-profile
  execution.
- [`qp_maros`](qp_maros): Maros--Meszaros FA_CP output normalization.

Each subdirectory documents its required inputs and command-line entry points.
Write generated data and solver outputs to local paths outside version control.
