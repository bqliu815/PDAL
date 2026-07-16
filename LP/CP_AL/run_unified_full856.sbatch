#!/bin/bash
#SBATCH -J cpal_uni_856
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 07:00:00
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then source /etc/profile >/dev/null 2>&1 || true; fi
module purge 2>/dev/null || true
CUDA_MODULE=${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}
if [[ -n "$CUDA_MODULE" ]]; then
  module load "$CUDA_MODULE" 2>/dev/null || true
fi

MANIFEST=${MANIFEST:?set MANIFEST to the generated full856 TSV}
BIN=${BIN:?set BIN to the CP_AL executable}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
EXPECTED_BINARY_SHA256=${EXPECTED_BINARY_SHA256:-}
EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256:-}
ITER_LIM=${ITER_LIM:-2147483647}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ -z "${SRUN:-}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    SRUN=$(command -v srun)
  elif [[ -x /opt/gridview/slurm/bin/srun ]]; then
    SRUN=/opt/gridview/slurm/bin/srun
  else
    echo "Cannot find srun; set SRUN to the Slurm launcher" >&2
    exit 127
  fi
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  else
    echo "ERROR: set CUDA_HOME or make nvcc available on PATH" >&2
    exit 127
  fi
fi

actual_binary_sha=$(sha256sum "$BIN" | awk '{print $1}')
actual_manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
if [[ -n "$EXPECTED_BINARY_SHA256" && "$actual_binary_sha" != "$EXPECTED_BINARY_SHA256" ]]; then
  echo "Binary SHA256 mismatch: expected $EXPECTED_BINARY_SHA256, found $actual_binary_sha" >&2
  exit 3
fi
if [[ -n "$EXPECTED_MANIFEST_SHA256" && "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "Manifest SHA256 mismatch: expected $EXPECTED_MANIFEST_SHA256, found $actual_manifest_sha" >&2
  exit 3
fi

line=$(awk -F '\t' -v target=$((TASK_ID + 2)) 'NR == target {print; exit}' "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "No manifest row for array task $TASK_ID" >&2
  exit 2
fi
IFS=$'\t' read -r dataset split instance tolerance time_limit mps <<< "$line"
mps=${mps%$'\r'}
if [[ ! -r "$mps" ]]; then
  echo "Input file is not readable: $mps" >&2
  exit 2
fi

task_dir="$OUT_ROOT/task_$(printf '%03d' "$TASK_ID")_${dataset}_${instance}_${tolerance}"
solver_dir="$task_dir/solver_output"
mkdir -p "$solver_dir"

while IFS='=' read -r name _; do
  if [[ "$name" == CP_AL_* ]]; then
    unset "$name"
  fi
done < <(env)

# Unified identity-free controller: one switch, resolved inside the binary.
# The binary computes anonymous matrix features and self-selects the route
#   (lowobj_highcv_eval100_cr5 / zero_equality_cr0 /
#    high_equality_tail_or_base).
# It reads no instance name, path, dataset, split, tolerance, or manifest row.
# ALPHA_DIRECT_PULSE_TRACE is enabled only for auditing positive-sigma events;
# it does not itself trigger any pulse (the tail route enables its own pulse).
export CP_AL_UNIFIED_POLICY=1
export CP_AL_ALPHA_DIRECT_PULSE_TRACE=1
export CP_AL_PROGRESS_LOG_FREQ=0
LD_PATH="$(dirname "$BIN")/lib:$(dirname "$BIN")"
if [[ -n "${CUDA_HOME:-}" ]]; then
  LD_PATH="$LD_PATH:$CUDA_HOME/lib64"
fi
export LD_LIBRARY_PATH="$LD_PATH:${LD_LIBRARY_PATH:-}"

if ldd "$BIN" | grep -q "not found"; then
  echo "Unresolved shared libraries for $BIN:" >&2
  ldd "$BIN" | grep "not found" >&2
  exit 127
fi

printf 'dataset\tsplit\tinstance\ttolerance\ttime_limit\tmps\n%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$dataset" "$split" "$instance" "$tolerance" "$time_limit" "$mps" > "$task_dir/task.tsv"
hostname > "$task_dir/node.txt"
printf '%s  %s\n' "$actual_binary_sha" "$BIN" > "$task_dir/binary.sha256"
printf '%s  %s\n' "$actual_manifest_sha" "$MANIFEST" > "$task_dir/manifest.sha256"

start=$(date +%s.%N)
set +e
"$SRUN" --ntasks=1 --unbuffered "$BIN" \
  --time_limit "$time_limit" --iter_limit "$ITER_LIM" \
  --eps_opt "$tolerance" --eps_feas "$tolerance" --eval_freq 200 --no_presolve \
  "$mps" "$solver_dir" > "$task_dir/solver.log" 2>&1
code=$?
set -e
end=$(date +%s.%N)

python3 - "$task_dir/wrapper.json" "$start" "$end" "$code" <<'PY'
import json
import sys

path, start, end, code = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"wall_seconds": float(end) - float(start), "exit_code": int(code)},
        handle,
        indent=2,
    )
    handle.write("\n")
PY

exit "$code"
