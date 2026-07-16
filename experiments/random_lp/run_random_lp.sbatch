#!/bin/bash
#SBATCH -J random_lp
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

set -uo pipefail

MANIFEST=${MANIFEST:?set MANIFEST to suite_manifest.tsv}
SUITE_DIR=${SUITE_DIR:?set SUITE_DIR to the generated suite directory}
CPAL_BIN=${CPAL_BIN:?set CPAL_BIN to the released CP_AL executable}
CUPDLPX_BIN=${CUPDLPX_BIN:?set CUPDLPX_BIN to the cuPDLPx executable}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256:-}
EXPECTED_CPAL_SHA256=${EXPECTED_CPAL_SHA256:-}
EXPECTED_CUPDLPX_SHA256=${EXPECTED_CUPDLPX_SHA256:-}
GROUP_SIZE=${GROUP_SIZE:-10}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TIME_LIMIT=${TIME_LIMIT:-60}
ITER_LIMIT=${ITER_LIMIT:-100000}
TOLERANCE=${TOLERANCE:-1e-8}
EVAL_FREQ=${EVAL_FREQ:-20}

if ! command -v module >/dev/null 2>&1; then source /etc/profile >/dev/null 2>&1 || true; fi
module purge >/dev/null 2>&1 || true
module load "${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}" >/dev/null 2>&1 || true

if [[ -z "${SRUN:-}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    SRUN=$(command -v srun)
  elif [[ -x /opt/gridview/slurm/bin/srun ]]; then
    SRUN=/opt/gridview/slurm/bin/srun
  else
    echo "Cannot find srun; set SRUN explicitly" >&2
    exit 127
  fi
fi

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
manifest_sha=$(sha256_file "$MANIFEST")
cpal_sha=$(sha256_file "$CPAL_BIN")
cupdlpx_sha=$(sha256_file "$CUPDLPX_BIN")

check_hash() {
  local label=$1 expected=$2 actual=$3
  if [[ -n "$expected" && "$expected" != "$actual" ]]; then
    echo "$label SHA256 mismatch: expected $expected, found $actual" >&2
    exit 3
  fi
}
check_hash manifest "$EXPECTED_MANIFEST_SHA256" "$manifest_sha"
check_hash CP_AL "$EXPECTED_CPAL_SHA256" "$cpal_sha"
check_hash cuPDLPx "$EXPECTED_CUPDLPX_SHA256" "$cupdlpx_sha"

mkdir -p "$OUT_ROOT"
start_row=$((TASK_ID * GROUP_SIZE + 2))
end_row=$((start_row + GROUP_SIZE - 1))
failed=0

clear_cpal_environment() {
  while IFS='=' read -r name _; do
    if [[ "$name" == CP_AL_* ]]; then unset "$name"; fi
  done < <(env)
}

run_solver() {
  local method=$1 bin=$2 instance=$3 mps=$4
  local method_dir="$OUT_ROOT/runs/$instance/$method"
  local solver_dir="$method_dir/solver_output"
  mkdir -p "$solver_dir"

  clear_cpal_environment
  if [[ "$method" == "rhr_cpal" ]]; then
    # Paper protocol: sigma_0 = 0.5 lambda^{-1/2} and
    # Reliable epochs use sigma^+ = sqrt(sigma * sigma_hat); unreliable
    # displacement estimates revert to the retained safeguarded value.
    export CP_AL_SIGMA_MODE=1
    export CP_AL_SIGMA_ADAPTIVE=1
    export CP_AL_SIGMA_INIT_FACTOR=0.5
    export CP_AL_SIGMA_MIN_FACTOR=1e-4
    export CP_AL_SIGMA_MAX_FACTOR=100
    export CP_AL_SIGMA_BALANCE_GAIN=0
    export CP_AL_SIGMA_KP=0.5
    export CP_AL_SIGMA_KI=0
    export CP_AL_SIGMA_KD=0
    export CP_AL_SIGMA_UPDATE_MIN=1e-6
    export CP_AL_SIGMA_UPDATE_MAX=1e6
    export CP_AL_LAMBDA_FACTOR=2.01
    export CP_AL_LAMBDA=0
    export CP_AL_RESTART_SUFFICIENT=0.2
    export CP_AL_RESTART_NECESSARY=0.8
    export CP_AL_RESTART_ARTIFICIAL=0.36
    export CP_AL_PROGRESS_LOG_FREQ=0
  fi

  printf '%s  %s\n' "$manifest_sha" "$MANIFEST" > "$method_dir/manifest.sha256"
  if [[ "$method" == "rhr_cpal" ]]; then
    printf '%s  %s\n' "$cpal_sha" "$bin" > "$method_dir/binary.sha256"
  else
    printf '%s  %s\n' "$cupdlpx_sha" "$bin" > "$method_dir/binary.sha256"
  fi
  hostname > "$method_dir/node.txt"

  local begin finish code
  begin=$(date +%s.%N)
  "$SRUN" --ntasks=1 --unbuffered "$bin" \
    --time_limit "$TIME_LIMIT" --iter_limit "$ITER_LIMIT" \
    --eps_opt "$TOLERANCE" --eps_feas "$TOLERANCE" \
    --eval_freq "$EVAL_FREQ" --no_presolve \
    "$mps" "$solver_dir" </dev/null > "$method_dir/solver.log" 2>&1
  code=$?
  finish=$(date +%s.%N)

  python3 - "$method_dir/wrapper.json" "$begin" "$finish" "$code" <<'PY'
import json
import sys

path, begin, finish, code = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"wall_seconds": float(finish) - float(begin), "exit_code": int(code)},
        handle,
        indent=2,
    )
    handle.write("\n")
PY
  if [[ $code -ne 0 ]]; then failed=1; fi
}

while IFS=$'\t' read -r family size instance seed m n nnz density mps_rel; do
  [[ -n "$instance" ]] || continue
  mps="$SUITE_DIR/${mps_rel%$'\r'}"
  if [[ ! -r "$mps" ]]; then
    echo "Missing generated instance: $mps" >&2
    failed=1
    continue
  fi
  mkdir -p "$OUT_ROOT/runs/$instance"
  printf 'family\tsize\tinstance\tseed\tm\tn\tnnz\tdensity_realized\tmps\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$family" "$size" "$instance" "$seed" "$m" "$n" "$nnz" "$density" "$mps" \
    > "$OUT_ROOT/runs/$instance/task.tsv"
  run_solver cupdlpx "$CUPDLPX_BIN" "$instance" "$mps"
  run_solver rhr_cpal "$CPAL_BIN" "$instance" "$mps"
done < <(sed -n "${start_row},${end_row}p" "$MANIFEST")

exit "$failed"
