#!/bin/bash
#SBATCH -J facp_qp134_arr
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 00:40:00
#SBATCH -o facp_qp134_arr-%A_%a.out
#SBATCH -e facp_qp134_arr-%A_%a.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE" || true
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

RUN_NAME="${RUN_NAME:-facp_restartselect_table1_134_array}"
RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/results_${RUN_NAME}/${RUN_TAG}}"
TIME_LIM="${TIME_LIM:-1000}"
ITER_LIM="${ITER_LIM:-2147483647}"
EPS_OPT="${EPS_OPT:-1e-6}"
EPS_FEAS="${EPS_FEAS:-1e-6}"
EVAL_FREQ="${EVAL_FREQ:-200}"
VERBOSE="${VERBOSE:-0}"
FA_CP_USE_AUGMENTATION="${FA_CP_USE_AUGMENTATION:-1}"
FA_CP_SIGMA="${FA_CP_SIGMA:-0}"
FA_CP_SIGMA_UPDATE="${FA_CP_SIGMA_UPDATE:-guarded}"
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

mapfile -d '' MPS_FILES < <(find "$DATA_DIR" -maxdepth 1 -type f \
  \( -iname "*.mps" -o -iname "*.qps" -o -iname "*.mps.gz" \) -print0 | sort -z)

N="${#MPS_FILES[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$N" -eq 0 ]]; then
  echo "ERROR: no instances selected."
  exit 1
fi
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "$N" ]]; then
  echo "ERROR: array task $TASK_ID outside [0,$((N - 1))]"
  exit 1
fi

LOG_DIR="$OUT_ROOT/logs"
ROW_DIR="$OUT_ROOT/rows"
SOLVER_OUT_DIR="$OUT_ROOT/solver_outputs"
mkdir -p "$LOG_DIR" "$ROW_DIR" "$SOLVER_OUT_DIR"

SUMMARY_CSV="$OUT_ROOT/summary.csv"
SUMMARY_HEADER="instance,status,optimal,termination_reason,exit_code,elapsed_sec,runtime_sec,total_iters,rel_primal,rel_dual,rel_gap,primal_obj,dual_obj,final_al_sigma,sigma_update_rule,sigma_update_count,mps,output_dir,summary_file,log"

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
else
  EXTRA_ARR=()
fi

mps="${MPS_FILES[$TASK_ID]}"
rel="$(basename "$mps")"
key="${rel%.*}"
out_dir="$SOLVER_OUT_DIR/$key"
mkdir -p "$out_dir"
log_file="$LOG_DIR/${key}.log"
row_file="$ROW_DIR/${TASK_ID}_${key}.csv"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "TASK_ID=$TASK_ID/$N"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "BUILD_DIR=$BUILD_DIR"
echo "BIN=$BIN"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "INSTANCE=$rel"

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
  cmd+=("--sigma_update" "$FA_CP_SIGMA_UPDATE")
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
  "$mps" "$out_dir" "${summary_file:-NA}" "$log_file" > "$row_file"

{
  flock -x 200
  if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "$SUMMARY_HEADER" > "$SUMMARY_CSV"
  fi
  cat "$row_file" >> "$SUMMARY_CSV"
} 200>"$SUMMARY_CSV.lock"

echo "$rel -> $status (term=$term, iters=$iter_total, code=$code, ${elapsed}s)"
