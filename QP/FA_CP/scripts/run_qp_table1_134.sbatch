#!/bin/bash
#SBATCH -J fa_cp_qp134
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o fa_cp_qp134-%j.out
#SBATCH -e fa_cp_qp134-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

CUDA_WHEEL_ROOT="${CUDA_WHEEL_ROOT:-}"
if [[ -d "$CUDA_WHEEL_ROOT" ]]; then
  export LD_LIBRARY_PATH="$CUDA_WHEEL_ROOT/cuda_runtime/lib:$CUDA_WHEEL_ROOT/cublas/lib:$CUDA_WHEEL_ROOT/cusparse/lib:$CUDA_WHEEL_ROOT/nvjitlink/lib:$CUDA_WHEEL_ROOT/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"
BIN="${BIN:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/../Maros-Meszaros-Table1-134}"

RUN_NAME="${RUN_NAME:-fa_cp_table1_134}"
TIME_LIM="${TIME_LIM:-1000}"
ITER_LIM="${ITER_LIM:-2147483647}"
EPS_OPT="${EPS_OPT:-1e-6}"
EPS_FEAS="${EPS_FEAS:-1e-6}"
EVAL_FREQ="${EVAL_FREQ:-200}"
VERBOSE="${VERBOSE:-0}"
MAX_CASES="${MAX_CASES:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
FA_CP_USE_AUGMENTATION="${FA_CP_USE_AUGMENTATION:-1}"
FA_CP_SIGMA="${FA_CP_SIGMA:-0}"
FA_CP_SIGMA_UPDATE="${FA_CP_SIGMA_UPDATE:-guarded}"
FA_CP_SIGMA_UPDATE_GAIN="${FA_CP_SIGMA_UPDATE_GAIN:-0.5}"
export FA_CP_RESTART_SELECT="${FA_CP_RESTART_SELECT:-0}"
SIGMA_CLIP_LOW="${SIGMA_CLIP_LOW:-0.6}"
SIGMA_CLIP_HIGH="${SIGMA_CLIP_HIGH:-1.7}"
SIGMA_MIN="${SIGMA_MIN:-1e-8}"
SIGMA_MAX="${SIGMA_MAX:-10}"
SIGMA_ROLLBACK_TOL="${SIGMA_ROLLBACK_TOL:-1.25}"
SIGMA_ZERO_THRESHOLD="${SIGMA_ZERO_THRESHOLD:-1e-8}"
SIGMA_ACTIVATION_THRESHOLD="${SIGMA_ACTIVATION_THRESHOLD:-2}"
SIGMA_NORMAL_MOTION_THRESHOLD="${SIGMA_NORMAL_MOTION_THRESHOLD:-1e-10}"
SIGMA_ACTIVATION_VALUE="${SIGMA_ACTIVATION_VALUE:-0.1}"
INSTANCE_LIST="${INSTANCE_LIST:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$BIN" ]]; then
  if [[ -x "$BUILD_DIR/fa_cp" ]]; then
    BIN="$BUILD_DIR/fa_cp"
  elif [[ -x "$BUILD_DIR/bin/fa_cp" ]]; then
    BIN="$BUILD_DIR/bin/fa_cp"
  else
    BIN="$BUILD_DIR/fa_cp"
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN"
  echo "Checked default locations:"
  echo "  $BUILD_DIR/fa_cp"
  echo "  $BUILD_DIR/bin/fa_cp"
  exit 1
fi
SRUN="${SRUN:-$(command -v srun || true)}"
if [[ -z "$SRUN" || ! -x "$SRUN" ]]; then
  echo "ERROR: srun not found. Set SRUN or load the Slurm environment."
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
echo "instance,status,optimal,termination_reason,exit_code,elapsed_sec,runtime_sec,total_iters,rel_primal,rel_dual,rel_gap,primal_obj,dual_obj,final_al_sigma,sigma_update_rule,sigma_update_count,mps,output_dir,summary_file,log" > "$SUMMARY_CSV"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "BIN=$BIN"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "FA_CP_USE_AUGMENTATION=$FA_CP_USE_AUGMENTATION"
echo "FA_CP_SIGMA=$FA_CP_SIGMA"
echo "FA_CP_SIGMA_UPDATE=$FA_CP_SIGMA_UPDATE"
echo "FA_CP_SIGMA_UPDATE_GAIN=$FA_CP_SIGMA_UPDATE_GAIN"
echo "FA_CP_RESTART_SELECT=$FA_CP_RESTART_SELECT"
echo "SIGMA_CLIP_LOW=$SIGMA_CLIP_LOW"
echo "SIGMA_CLIP_HIGH=$SIGMA_CLIP_HIGH"
echo "SIGMA_MIN=$SIGMA_MIN"
echo "SIGMA_MAX=$SIGMA_MAX"
echo "SIGMA_ROLLBACK_TOL=$SIGMA_ROLLBACK_TOL"
echo "SIGMA_ZERO_THRESHOLD=$SIGMA_ZERO_THRESHOLD"
echo "SIGMA_ACTIVATION_THRESHOLD=$SIGMA_ACTIVATION_THRESHOLD"
echo "SIGMA_NORMAL_MOTION_THRESHOLD=$SIGMA_NORMAL_MOTION_THRESHOLD"
echo "SIGMA_ACTIVATION_VALUE=$SIGMA_ACTIVATION_VALUE"
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

  if [[ "$FA_CP_USE_AUGMENTATION" == "1" ]]; then
    cmd+=("--al_sigma" "$FA_CP_SIGMA")
    if [[ "$FA_CP_SIGMA_UPDATE" == "fixed" || "$FA_CP_SIGMA_UPDATE" == "none" ]]; then
      cmd+=("--sigma_update" "$FA_CP_SIGMA_UPDATE")
    else
      cmd+=("--sigma_update" "$FA_CP_SIGMA_UPDATE"
            "--sigma_update_gain" "$FA_CP_SIGMA_UPDATE_GAIN"
            "--sigma_clip_low" "$SIGMA_CLIP_LOW"
            "--sigma_clip_high" "$SIGMA_CLIP_HIGH"
            "--sigma_min" "$SIGMA_MIN"
            "--sigma_max" "$SIGMA_MAX"
            "--sigma_rollback_tol" "$SIGMA_ROLLBACK_TOL"
            "--sigma_zero_threshold" "$SIGMA_ZERO_THRESHOLD"
            "--sigma_activation_threshold" "$SIGMA_ACTIVATION_THRESHOLD"
            "--sigma_normal_motion_threshold" "$SIGMA_NORMAL_MOTION_THRESHOLD"
            "--sigma_activation_value" "$SIGMA_ACTIVATION_VALUE")
    fi
  else
    cmd+=("--no_al_qp")
  fi
  if [[ "${#EXTRA_ARR[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARR[@]}")
  fi

  set +e
  "$SRUN" --ntasks=1 --unbuffered "${cmd[@]}" > "$log_file" 2>&1
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
  runtime_sec="NA"
  final_al_sigma="NA"
  sigma_update_rule="NA"
  sigma_update_count="NA"
  optimal="0"

  if [[ -n "${summary_file:-}" && -f "$summary_file" ]]; then
    term="$(grep -E '^Termination Reason:' "$summary_file" | tail -n1 | sed -E 's/^Termination Reason:[[:space:]]*//' || true)"
    runtime_sec="$(grep -F 'Runtime (sec):' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    iter_total="$(grep -E '^Iterations Count:' "$summary_file" | tail -n1 | sed -E 's/^Iterations Count:[[:space:]]*([0-9]+).*/\1/' || true)"
    rel_primal="$(grep -E '^Relative Primal Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    rel_dual="$(grep -E '^Relative Dual Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    rel_gap="$(grep -E '^Relative Objective Gap:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    primal_obj="$(grep -E '^Primal Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    dual_obj="$(grep -E '^Dual Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    final_al_sigma="$(grep -E '^Final AL Sigma:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    sigma_update_rule="$(grep -E '^Sigma Update Rule:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    sigma_update_count="$(grep -E '^Sigma Update Count:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)"
    if [[ "$term" == "OPTIMAL" ]]; then
      optimal="1"
    fi
  fi

  printf "%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$key" "$status" "$optimal" "$term" "$code" "$elapsed" \
    "${runtime_sec:-NA}" "$iter_total" \
    "${rel_primal:-NA}" "${rel_dual:-NA}" "${rel_gap:-NA}" \
    "${primal_obj:-NA}" "${dual_obj:-NA}" "${final_al_sigma:-NA}" \
    "${sigma_update_rule:-NA}" "${sigma_update_count:-NA}" \
    "$mps" "$out_dir" "${summary_file:-NA}" "$log_file" >> "$SUMMARY_CSV"

  echo "[$idx/$N] $rel -> $status (term=$term, iters=$iter_total, code=$code, ${elapsed}s)"

  if [[ "$code" -ne 0 && "$STOP_ON_ERROR" == "1" ]]; then
    echo "STOP_ON_ERROR=1, abort."
    exit "$code"
  fi
done

echo "Done. Summary: $SUMMARY_CSV"
