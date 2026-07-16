#!/bin/bash
# Run one modulo group with the base feature-policy controls.
# This diagnostic helper does not reproduce the paper's unified 856-task row;
# use run_unified_full856.sbatch for that result.

#SBATCH -J cp_al_mip
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o cp_al_mip-%j.out
#SBATCH -e cp_al_mip-%j.err

set -euo pipefail

MODULE_INIT="${MODULE_INIT:-}"
if ! command -v module >/dev/null 2>&1 && [[ -n "$MODULE_INIT" && -f "$MODULE_INIT" ]]; then
  source "$MODULE_INIT"
fi
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "${MODULE_USE_PATH:-}" ]]; then
    module use "$MODULE_USE_PATH" || true
  fi
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
BUILD_DIR=${BUILD_DIR:-$PROJECT_DIR/build}
BIN=${BIN:-$BUILD_DIR/cp_al}
DATASET_ROOT=${DATASET_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)/MIPLIB2017_gurobifinish}
SPLIT=${SPLIT:-Small}
GROUP_IDX=${GROUP_IDX:-0}
GROUP_COUNT=${GROUP_COUNT:-20}
PROFILE=${PROFILE:-CP_AL_feature_cr_safe0}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/results_${PROFILE}}

LD_PATH_PARTS="$BUILD_DIR/lib:$BUILD_DIR"
if [[ -n "${HIGHS_HOME:-}" ]]; then
  LD_PATH_PARTS="$LD_PATH_PARTS:$HIGHS_HOME/lib"
fi
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")}
  LD_PATH_PARTS="$LD_PATH_PARTS:$CUDA_HOME/lib64"
fi
export LD_LIBRARY_PATH="$LD_PATH_PARTS:${LD_LIBRARY_PATH:-}"

TIME_LIM=${TIME_LIM:-3600}
ITER_LIM=${ITER_LIM:-2147483647}
EPS_OPT=${EPS_OPT:-1e-8}
EPS_FEAS=${EPS_FEAS:-1e-8}
EVAL_FREQ=${EVAL_FREQ:-200}
VERBOSE=${VERBOSE:-0}
NO_PRESOLVE=${NO_PRESOLVE:-1}
export CP_AL_FEATURE_AUTO_POLICY="${CP_AL_FEATURE_AUTO_POLICY:-1}"
export CP_AL_FEATURE_POLICY_VARIANT="${CP_AL_FEATURE_POLICY_VARIANT:-safe0}"

DATA_DIR="$DATASET_ROOT/$SPLIT"
GROUP_DIR="$OUT_ROOT/$SPLIT/group_$(printf '%02d' "$GROUP_IDX")"
LOG_DIR="$GROUP_DIR/logs"
SOLVER_OUT_DIR="$GROUP_DIR/solver_outputs"
mkdir -p "$LOG_DIR" "$SOLVER_OUT_DIR"

SUMMARY_CSV="$GROUP_DIR/summary.csv"
echo "solver,split,group_idx,group_count,instance,status,optimal,termination_reason,exit_code,elapsed_sec,runtime_sec,precondition_sec,policy_sec,total_internal_sec,total_algorithm_sec,converged_iters,total_iters,final_lambda,lambda_events,restart_select_events,restart_select_average,restart_select_halpern,restart_select_current,mps,output_dir,summary_file,log" > "$SUMMARY_CSV"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NOTE: diagnostic base-policy runner; not the released unified benchmark entry point"
echo "PROFILE=$PROFILE"
echo "BIN=$BIN"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "GROUP_IDX=$GROUP_IDX GROUP_COUNT=$GROUP_COUNT"
echo "NO_PRESOLVE=$NO_PRESOLVE"
echo "CP_AL_FEATURE_AUTO_POLICY=$CP_AL_FEATURE_AUTO_POLICY"
echo "CP_AL_FEATURE_POLICY_VARIANT=$CP_AL_FEATURE_POLICY_VARIANT"
echo "CP_AL_LAMBDA=${CP_AL_LAMBDA:-unset}"
echo "CP_AL_LAMBDA_PROFILE=${CP_AL_LAMBDA_PROFILE:-unset}"
echo "CP_AL_LAMBDA_GRID=${CP_AL_LAMBDA_GRID:-unset}"
echo "CP_AL_ADAPTIVE_LAMBDA=${CP_AL_ADAPTIVE_LAMBDA:-unset}"
echo "CP_AL_LAMBDA_BASE=${CP_AL_LAMBDA_BASE:-unset}"
echo "CP_AL_LAMBDA_LOW=${CP_AL_LAMBDA_LOW:-unset}"
echo "CP_AL_LAMBDA_MID=${CP_AL_LAMBDA_MID:-unset}"
echo "CP_AL_LAMBDA_HIGH=${CP_AL_LAMBDA_HIGH:-unset}"
echo "CP_AL_LAMBDA_RACE=${CP_AL_LAMBDA_RACE:-unset}"
echo "CP_AL_LAMBDA_FORCE_RESTART_ITERS=${CP_AL_LAMBDA_FORCE_RESTART_ITERS:-unset}"
echo "CP_AL_LAMBDA_MIN_RESTARTS=${CP_AL_LAMBDA_MIN_RESTARTS:-unset}"
echo "CP_AL_LAMBDA_MIN_ITERS=${CP_AL_LAMBDA_MIN_ITERS:-unset}"
echo "CP_AL_RESTART_SELECT=${CP_AL_RESTART_SELECT:-unset}"
echo "CP_AL_RESTART_SELECT_PROFILE=${CP_AL_RESTART_SELECT_PROFILE:-unset}"
echo "CP_AL_RESTART_SELECT_MARGIN=${CP_AL_RESTART_SELECT_MARGIN:-unset}"
echo "CP_AL_RESTART_SELECT_MIN_TIME=${CP_AL_RESTART_SELECT_MIN_TIME:-unset}"
echo "CP_AL_RESTART_SELECT_MIN_ITERS=${CP_AL_RESTART_SELECT_MIN_ITERS:-unset}"
echo "CP_AL_RESTART_SELECT_TRACE=${CP_AL_RESTART_SELECT_TRACE:-unset}"
echo "CP_AL_RESTART_SUFFICIENT=${CP_AL_RESTART_SUFFICIENT:-unset}"
echo "CP_AL_RESTART_NECESSARY=${CP_AL_RESTART_NECESSARY:-unset}"
echo "CP_AL_RESTART_ARTIFICIAL=${CP_AL_RESTART_ARTIFICIAL:-unset}"
echo "CP_AL_SIGMA_MODE=${CP_AL_SIGMA_MODE:-unset}"
echo "CP_AL_SIGMA_TRIGGER=${CP_AL_SIGMA_TRIGGER:-unset}"

MPS_FILES=()
while IFS= read -r -d '' f; do
  MPS_FILES+=("$f")
done < <(find "$DATA_DIR" -maxdepth 1 -type f \( -name "*.mps" -o -name "*.MPS" -o -name "*.mps.gz" -o -name "*.MPS.GZ" \) -print0 | sort -z)

N=${#MPS_FILES[@]}
if [[ "$N" -eq 0 ]]; then
  echo "ERROR: no MPS files found in $DATA_DIR"
  exit 1
fi

idx=0
for mps in "${MPS_FILES[@]}"; do
  mod=$((idx % GROUP_COUNT))
  idx=$((idx + 1))
  if [[ "$mod" -ne "$GROUP_IDX" ]]; then
    continue
  fi

  base=$(basename "$mps")
  inst="${base%.*}"
  if [[ "$base" == *.mps.gz || "$base" == *.MPS.GZ ]]; then
    inst="${base%.*.*}"
  fi
  key_sanitized=$(echo "$inst" | sed -E 's#[ /]+#_#g')
  out_dir="$SOLVER_OUT_DIR/$key_sanitized"
  log_file="$LOG_DIR/${key_sanitized}.log"
  mkdir -p "$out_dir"

  cmd=("$BIN"
       "--time_limit" "$TIME_LIM"
       "--iter_limit" "$ITER_LIM"
       "--eps_opt" "$EPS_OPT"
       "--eps_feas" "$EPS_FEAS"
       "--eval_freq" "$EVAL_FREQ")
  if [[ "$VERBOSE" == "1" ]]; then
    cmd+=("--verbose")
  fi
  if [[ "$NO_PRESOLVE" == "1" ]]; then
    cmd+=("--no_presolve")
  fi
  cmd+=("$mps" "$out_dir")

  echo "Running $SPLIT/$inst"
  start_ts=$(date +%s)
  set +e
  "${cmd[@]}" > "$log_file" 2>&1
  code=$?
  set -e
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  status="ok"
  if [[ "$code" -ne 0 ]]; then
    status="fail"
  fi

  term=$(grep -E "Status[[:space:]]*:" "$log_file" | tail -n1 | sed -E 's/^.*Status[[:space:]]*:[[:space:]]*//' || true)
  runtime=$(grep -E "Solve time[[:space:]]*:" "$log_file" | tail -n1 | sed -E 's/^.*Solve time[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/' || true)
  precondition=$(grep -E "Precondition time[[:space:]]*:" "$log_file" | tail -n1 | sed -E 's/^.*Precondition time[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/' || true)
  policy=$(grep -E "\[feature-auto-policy-time\]" "$log_file" | tail -n1 | sed -E 's/^.*seconds=([0-9.eE+-]+).*$/\1/' || true)
  iters=$(grep -E "Iterations[[:space:]]*:" "$log_file" | tail -n1 | sed -E 's/^.*Iterations[[:space:]]*:[[:space:]]*([0-9]+).*/\1/' || true)
  summary_file=$(find "$out_dir" -maxdepth 1 -type f -name "*_summary.txt" | head -n1 || true)
  if [[ -z "${term:-}" && -n "${summary_file:-}" && -f "$summary_file" ]]; then
    term=$(grep -E "^Termination Reason:" "$summary_file" | tail -n1 | sed -E 's/^Termination Reason:[[:space:]]*//' || true)
  fi
  if [[ -z "${iters:-}" && -n "${summary_file:-}" && -f "$summary_file" ]]; then
    iters=$(grep -E "^Iterations Count:" "$summary_file" | tail -n1 | sed -E 's/^Iterations Count:[[:space:]]*([0-9]+).*/\1/' || true)
  fi
  if [[ -z "${runtime:-}" && -n "${summary_file:-}" && -f "$summary_file" ]]; then
    runtime=$(grep -E "^(Solve Time|Solve time|Runtime)" "$summary_file" | tail -n1 | sed -E 's/^.*:[[:space:]]*([0-9.eE+-]+).*/\1/' || true)
  fi
  total_internal="NA"
  if [[ -n "${runtime:-}" && -n "${precondition:-}" ]]; then
    total_internal=$(awk -v solve="$runtime" -v prep="$precondition" 'BEGIN { printf "%.12g", solve + prep }')
  fi
  total_algorithm="NA"
  if [[ -n "${runtime:-}" && -n "${precondition:-}" && -n "${policy:-}" ]]; then
    total_algorithm=$(awk -v solve="$runtime" -v prep="$precondition" -v policy="$policy" \
      'BEGIN { printf "%.12g", solve + prep + policy }')
  fi

  optimal=0
  converged_iters="NA"
  if [[ "${term:-NA}" == "OPTIMAL" ]]; then
    optimal=1
    converged_iters="${iters:-NA}"
  fi

  final_lambda=$(grep -E "lambda-(adapt|race) restart" "$log_file" | tail -n1 | sed -E 's/.*lambda [0-9.eE+-]+->([0-9.eE+-]+).*/\1/' || true)
  if [[ -z "${final_lambda:-}" ]]; then
    final_lambda=$(grep "\\[box-cpal\\]" "$log_file" | head -n1 | sed -E 's/.*al_lambda=([0-9.eE+-]+).*/\1/' || true)
  fi
  lambda_events=$(grep -cE "lambda-(adapt|race) restart" "$log_file" || true)
  restart_select_events=$(grep -cE "\\[restart-select\\]" "$log_file" || true)
  restart_select_average=$(grep -cE "\\[restart-select\\].*target=(average|reflected_halpern)" "$log_file" || true)
  restart_select_halpern=$(grep -cE "\\[restart-select\\].*target=halpern" "$log_file" || true)
  restart_select_current=$(grep -cE "\\[restart-select\\].*target=current" "$log_file" || true)

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$PROFILE" "$SPLIT" "$GROUP_IDX" "$GROUP_COUNT" "$inst" "$status" "$optimal" "${term:-NA}" "$code" "$elapsed" \
    "${runtime:-NA}" "${precondition:-NA}" "${policy:-NA}" "$total_internal" "$total_algorithm" "$converged_iters" "${iters:-NA}" "${final_lambda:-NA}" "$lambda_events" \
    "$restart_select_events" "$restart_select_average" "$restart_select_halpern" "$restart_select_current" \
    "$mps" "$out_dir" "${summary_file:-NA}" "$log_file" \
    >> "$SUMMARY_CSV"
  echo "$SPLIT/$inst -> $status term=${term:-NA} runtime=${runtime:-NA} iters=${iters:-NA}"
done

echo "Done. Summary: $SUMMARY_CSV"
