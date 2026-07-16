#!/bin/bash
#SBATCH -J rhr_lin_cpal_qp134
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o rhr_lin_cpal_qp134-%j.out
#SBATCH -e rhr_lin_cpal_qp134-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE" || true
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"
BIN="${BIN:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/../Maros-Meszaros-Table1-134}"

RUN_NAME="${RUN_NAME:-rhr_lin_cpal_qp_table1_134}"
TIME_LIM="${TIME_LIM:-1000}"
ITER_LIM="${ITER_LIM:-2147483647}"
EPS_OPT="${EPS_OPT:-1e-6}"
EPS_FEAS="${EPS_FEAS:-1e-6}"
EVAL_FREQ="${EVAL_FREQ:-200}"
VERBOSE="${VERBOSE:-0}"
MAX_CASES="${MAX_CASES:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
USE_AL_QP="${USE_AL_QP:-1}"
SIGMA_INIT="${SIGMA_INIT:-${AL_LAMBDA:-0}}"
INSTANCE_LIST="${INSTANCE_LIST:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$BIN" ]]; then
  if [[ -x "$BUILD_DIR/rhr_lin_cp_al" ]]; then
    BIN="$BUILD_DIR/rhr_lin_cp_al"
  elif [[ -x "$BUILD_DIR/bin/rhr_lin_cp_al" ]]; then
    BIN="$BUILD_DIR/bin/rhr_lin_cp_al"
  else
    BIN="$BUILD_DIR/rhr_lin_cp_al"
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN"
  echo "Checked default locations:"
  echo "  $BUILD_DIR/rhr_lin_cp_al"
  echo "  $BUILD_DIR/bin/rhr_lin_cp_al"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi
if [[ "$MAX_CASES" -lt 0 ]]; then
  echo "ERROR: MAX_CASES must be >= 0"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}_${TS}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/results_${RUN_NAME}/${RUN_TAG}}"
LOG_DIR="$OUT_ROOT/logs"
SOLVER_OUT_DIR="$OUT_ROOT/solver_outputs"
mkdir -p "$LOG_DIR" "$SOLVER_OUT_DIR"

SUMMARY_CSV="$OUT_ROOT/summary.csv"
echo "instance,status,optimal,termination_reason,exit_code,elapsed_sec,total_iters,rel_primal,rel_dual,rel_gap,primal_obj,dual_obj,mps,output_dir,summary_file,log" > "$SUMMARY_CSV"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "BIN=$BIN"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "USE_AL_QP=$USE_AL_QP"
echo "SIGMA_INIT=$SIGMA_INIT"
echo "INSTANCE_LIST=${INSTANCE_LIST:-<none>}"
echo "MAX_CASES=$MAX_CASES"

MPS_FILES=()
while IFS= read -r -d '' f; do
  MPS_FILES+=("$f")
done < <(find "$DATA_DIR" -maxdepth 1 -type f \( -iname "*.mps" -o -iname "*.qps" -o -iname "*.mps.gz" \) -print0 | sort -z)

if [[ -n "$INSTANCE_LIST" ]]; then
  if [[ ! -f "$INSTANCE_LIST" ]]; then
    echo "ERROR: INSTANCE_LIST not found: $INSTANCE_LIST"
    exit 1
  fi
  FILTERED=()
  while IFS= read -r raw; do
    line="${raw#"${raw%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    found=""
    if [[ -f "$line" ]]; then
      found="$line"
    else
      for cand in "${MPS_FILES[@]}"; do
        base="$(basename "$cand")"
        stem="${base%.*}"
        if [[ "$line" == "$base" || "$line" == "$stem" ]]; then
          found="$cand"
          break
        fi
      done
    fi
    if [[ -n "$found" ]]; then
      FILTERED+=("$found")
    else
      echo "WARN: skip unmatched instance: $line"
    fi
  done < "$INSTANCE_LIST"
  MPS_FILES=("${FILTERED[@]}")
fi

N="${#MPS_FILES[@]}"
if [[ "$N" -eq 0 ]]; then
  echo "ERROR: no instances selected."
  exit 1
fi

if [[ "$MAX_CASES" -gt 0 && "$MAX_CASES" -lt "$N" ]]; then
  N="$MAX_CASES"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
else
  EXTRA_ARR=()
fi

idx=0
for mps in "${MPS_FILES[@]}"; do
  idx=$((idx + 1))
  if [[ "$MAX_CASES" -gt 0 && "$idx" -gt "$MAX_CASES" ]]; then
    break
  fi

  rel="$(basename "$mps")"
  key="${rel%.*}"
  out_dir="$SOLVER_OUT_DIR/$key"
  mkdir -p "$out_dir"
  log_file="$LOG_DIR/${key}.log"

  echo "[$idx/$N] Running $rel"
  start_ts="$(date +%s)"

  cmd=("$BIN"
       "$mps" "$out_dir"
       "--time_limit" "$TIME_LIM"
       "--iter_limit" "$ITER_LIM"
       "--eps_opt" "$EPS_OPT"
       "--eps_feas" "$EPS_FEAS"
       "--eval_freq" "$EVAL_FREQ"
       "-v" "$VERBOSE")

  if [[ "$USE_AL_QP" == "1" ]]; then
    cmd+=("--sigma_init" "$SIGMA_INIT")
  else
    cmd+=("--no_al_qp")
  fi
  if [[ "${#EXTRA_ARR[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARR[@]}")
  fi

  set +e
  srun --ntasks=1 --unbuffered "${cmd[@]}" > "$log_file" 2>&1
  code="$?"
  set -e

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"

  status="ok"
  if [[ "$code" -ne 0 ]]; then
    status="fail"
  fi

  summary_file="$(find "$out_dir" -maxdepth 1 -type f -name '*_summary.txt' | head -n1 || true)"
  term="NA"
  iter_total="NA"
  rel_primal="NA"
  rel_dual="NA"
  rel_gap="NA"
  primal_obj="NA"
  dual_obj="NA"
  optimal="0"

  if [[ -n "${summary_file:-}" && -f "$summary_file" ]]; then
    term="$(grep -E '^Termination Reason:' "$summary_file" | tail -n1 | sed -E 's/^Termination Reason:[[:space:]]*//' || true)"
    iter_total="$(grep -E '^Iterations Count:' "$summary_file" | tail -n1 | sed -E 's/^Iterations Count:[[:space:]]*([0-9]+).*/\1/' || true)"
    rel_primal="$(grep -E '^Relative Primal Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    rel_dual="$(grep -E '^Relative Dual Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    rel_gap="$(grep -E '^Relative Objective Gap:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    primal_obj="$(grep -E '^Primal Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    dual_obj="$(grep -E '^Dual Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    if [[ "$term" == "OPTIMAL" ]]; then
      optimal="1"
    fi
  fi

  printf "%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$key" "$status" "$optimal" "$term" "$code" "$elapsed" "$iter_total" \
    "${rel_primal:-NA}" "${rel_dual:-NA}" "${rel_gap:-NA}" \
    "${primal_obj:-NA}" "${dual_obj:-NA}" \
    "$mps" "$out_dir" "${summary_file:-NA}" "$log_file" >> "$SUMMARY_CSV"

  echo "[$idx/$N] $rel -> $status (term=$term, iters=$iter_total, code=$code, ${elapsed}s)"

  if [[ "$code" -ne 0 && "$STOP_ON_ERROR" == "1" ]]; then
    echo "STOP_ON_ERROR=1, abort."
    exit "$code"
  fi
done

echo "Done. Summary: $SUMMARY_CSV"
