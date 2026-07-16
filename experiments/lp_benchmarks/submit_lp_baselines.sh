#!/bin/bash
# Copyright 2026 Benqi Liu
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
BASELINE_ROOT=${BASELINE_ROOT:?set BASELINE_ROOT to the third-party installation root}
MANIFEST=${MANIFEST:?set MANIFEST to the common full856.tsv}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a fresh output directory}
SBATCH=${SBATCH:-$(command -v sbatch 2>/dev/null || true)}
SOLVERS=${SOLVERS:-"cupdlpx_c cupdlpc cupdlp_jl hprlp_jl hprlp_c"}
MAX_PARALLEL_PER_ARRAY=${MAX_PARALLEL_PER_ARRAY:-8}
if [[ -z "$SBATCH" || ! -x "$SBATCH" ]]; then
  echo "sbatch executable not found; set SBATCH explicitly." >&2
  exit 2
fi
if [[ -z "${JULIA:-}" && -x "$BASELINE_ROOT/../software/julia-1.11.2/bin/julia" ]]; then
  JULIA="$BASELINE_ROOT/../software/julia-1.11.2/bin/julia"
fi
JULIA=${JULIA:-$(command -v julia 2>/dev/null || true)}

mkdir -p "$RUN_ROOT/cells" "$RUN_ROOT/submission_logs"
python3 "$REPRO_ROOT/experiments/lp_benchmarks/make_cell_manifests.py" \
  --manifest "$MANIFEST" --output-dir "$RUN_ROOT/cells"
sha256sum "$MANIFEST" > "$RUN_ROOT/full856_manifest.sha256"
{
  date -Iseconds
  if [[ -n "$JULIA" ]]; then "$JULIA" --version; else printf 'Julia not found\n'; fi
  sha256sum \
    "$BASELINE_ROOT/src/cuPDLPx-C-v0.2.9/build/cupdlpx" \
    "$BASELINE_ROOT/src/cuPDLP-C-v0.4.1/build_gpu/bin/plc" \
    "$BASELINE_ROOT/src/HPR-LP-C-v0.1.2/build/solve_mps_file" \
    "$REPRO_ROOT/experiments/lp_benchmarks/submit_lp_baselines.sh" \
    "$REPRO_ROOT/experiments/lp_benchmarks/run_baseline_group.sbatch" \
    "$REPRO_ROOT/experiments/lp_benchmarks/run_cupdlp_jl_cli.jl" \
    "$REPRO_ROOT/experiments/lp_benchmarks/run_hprlp_one.jl" \
    "$REPRO_ROOT/experiments/lp_benchmarks/make_cell_manifests.py" \
    "$REPRO_ROOT/experiments/lp_benchmarks/parse_outputs.py" \
    "$REPRO_ROOT/experiments/lp_benchmarks/collect_lp_tables.py"
  for source in \
    "$BASELINE_ROOT/src/cuPDLPx-C-v0.2.9" \
    "$BASELINE_ROOT/src/cuPDLP-C-v0.4.1" \
    "$BASELINE_ROOT/src/cuPDLP-jl-master" \
    "$BASELINE_ROOT/src/HPR-LP-v0.1.0" \
    "$BASELINE_ROOT/src/HPR-LP-C-v0.1.2"; do
    printf '%s\t' "$source"
    git -C "$source" rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout\n'
  done
} > "$RUN_ROOT/software_provenance.txt"

submit_cell() {
  local solver=$1 cell=$2 tolerance=$3 group_size=$4 wall=$5 memory=$6
  local tag="${cell}_eps${tolerance//-/m}" effective_memory=$memory
  local cell_manifest="$RUN_ROOT/cells/${tag}.tsv"
  local rows groups out result
  if [[ "$solver" == "hprlp_jl" ]]; then
    effective_memory=96G
  fi
  rows=$(( $(wc -l < "$cell_manifest") - 1 ))
  groups=$(( (rows + group_size - 1) / group_size ))
  out="$RUN_ROOT/${solver}_${tag}"
  mkdir -p "$out/logs"
  result=$(
    "$SBATCH" --parsable --job-name="${solver}_${cell}_${tolerance}" \
      --time="$wall" --mem="$effective_memory" \
      --array="0-$((groups - 1))%$MAX_PARALLEL_PER_ARRAY" \
      --output="$out/logs/slurm-%A_%a.out" --error="$out/logs/slurm-%A_%a.err" \
      --export="ALL,BASELINE_ROOT=$BASELINE_ROOT,REPRO_ROOT=$REPRO_ROOT,SOLVER=$solver,CELL_MANIFEST=$cell_manifest,OUT_ROOT=$out,GROUP_SIZE=$group_size,JULIA=$JULIA" \
      "$REPRO_ROOT/experiments/lp_benchmarks/run_baseline_group.sbatch"
  )
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$result" "$solver" "$cell" "$tolerance" "$effective_memory" "$out" \
    | tee -a "$RUN_ROOT/submissions.tsv"
}

printf 'job_id\tsolver\tcell\ttolerance\tmemory\toutput\n' > "$RUN_ROOT/submissions.tsv"
for solver in $SOLVERS; do
  for tolerance in 1e-4 1e-8; do
    submit_cell "$solver" Small "$tolerance" 6 08:00:00 32G
    submit_cell "$solver" Medium "$tolerance" 2 03:00:00 48G
    submit_cell "$solver" Large "$tolerance" 1 06:00:00 96G
    submit_cell "$solver" Mittelmann "$tolerance" 4 02:30:00 48G
  done
done

echo "Submitted baseline run: $RUN_ROOT"
