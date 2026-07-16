#!/usr/bin/env bash

set -euo pipefail

BIN=${BIN:?set BIN to the compiled profile executable}
INSTANCE_LIST=${INSTANCE_LIST:?set INSTANCE_LIST to one QP chunk}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to the chunk output directory}
RUN_NAME=${RUN_NAME:?set RUN_NAME to the frozen profile name}
TIME_LIM=${TIME_LIM:-60}
ITER_LIM=${ITER_LIM:-2147483647}
EPS_OPT=${EPS_OPT:-1e-5}
EPS_FEAS=${EPS_FEAS:-1e-5}
EVAL_FREQ=${EVAL_FREQ:-200}
VERBOSE=${VERBOSE:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: executable not found: $BIN" >&2
  exit 2
fi
if [[ ! -f "$INSTANCE_LIST" ]]; then
  echo "ERROR: instance list not found: $INSTANCE_LIST" >&2
  exit 2
fi

LOG_DIR="$OUT_ROOT/logs"
SOLVER_OUT_DIR="$OUT_ROOT/solver_outputs"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
mkdir -p "$LOG_DIR" "$SOLVER_OUT_DIR"
printf '%s\n' \
  'instance,status,optimal,termination_reason,exit_code,elapsed_sec,total_iters,rel_primal,rel_dual,rel_gap,primal_obj,dual_obj,mps,output_dir,summary_file,log' \
  > "$SUMMARY_CSV"

mapfile -t MPS_FILES < <(sed '/^[[:space:]]*$/d' "$INSTANCE_LIST")
if [[ "${#MPS_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no instances selected by $INSTANCE_LIST" >&2
  exit 2
fi
for mps in "${MPS_FILES[@]}"; do
  if [[ ! -f "$mps" ]]; then
    echo "ERROR: listed QP does not exist: $mps" >&2
    exit 2
  fi
done

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
else
  EXTRA_ARR=()
fi

if [[ -n "${SRUN:-}" ]]; then
  read -r -a LAUNCHER <<< "$SRUN"
elif command -v srun >/dev/null 2>&1; then
  LAUNCHER=(srun --ntasks=1 --unbuffered)
else
  LAUNCHER=()
fi

total=${#MPS_FILES[@]}
index=0
for mps in "${MPS_FILES[@]}"; do
  index=$((index + 1))
  filename=$(basename "$mps")
  key=${filename%.*}
  out_dir="$SOLVER_OUT_DIR/$key"
  log_file="$LOG_DIR/$key.log"
  mkdir -p "$out_dir"

  cmd=(
    "$BIN" "$mps" "$out_dir"
    --time_limit "$TIME_LIM"
    --iter_limit "$ITER_LIM"
    --eps_opt "$EPS_OPT"
    --eps_feas "$EPS_FEAS"
    --eval_freq "$EVAL_FREQ"
    -v "$VERBOSE"
  )
  if [[ "${#EXTRA_ARR[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARR[@]}")
  fi

  echo "[$index/$total] $RUN_NAME: $filename"
  start_ts=$(date +%s)
  set +e
  "${LAUNCHER[@]}" "${cmd[@]}" > "$log_file" 2>&1
  code=$?
  set -e
  elapsed=$(( $(date +%s) - start_ts ))

  status=ok
  [[ "$code" -eq 0 ]] || status=fail
  summary_file=$(find "$out_dir" -maxdepth 1 -type f -name '*_summary.txt' | head -n1 || true)
  term=NA
  iter_total=NA
  rel_primal=NA
  rel_dual=NA
  rel_gap=NA
  primal_obj=NA
  dual_obj=NA
  optimal=0
  if [[ -n "$summary_file" && -f "$summary_file" ]]; then
    term=$(grep -E '^Termination Reason:' "$summary_file" | tail -n1 | sed -E 's/^Termination Reason:[[:space:]]*//' || true)
    iter_total=$(grep -E '^Iterations Count:' "$summary_file" | tail -n1 | sed -E 's/^Iterations Count:[[:space:]]*([0-9]+).*/\1/' || true)
    rel_primal=$(grep -E '^Relative Primal Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)
    rel_dual=$(grep -E '^Relative Dual Residual:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)
    rel_gap=$(grep -E '^Relative Objective Gap:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)
    primal_obj=$(grep -E '^Primal Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)
    dual_obj=$(grep -E '^Dual Objective Value:' "$summary_file" | tail -n1 | awk -F': ' '{print $2}' || true)
    [[ "$term" == OPTIMAL ]] && optimal=1
  fi

  printf '%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$key" "$status" "$optimal" "$term" "$code" "$elapsed" "$iter_total" \
    "${rel_primal:-NA}" "${rel_dual:-NA}" "${rel_gap:-NA}" \
    "${primal_obj:-NA}" "${dual_obj:-NA}" "$mps" "$out_dir" \
    "${summary_file:-NA}" "$log_file" >> "$SUMMARY_CSV"

  if [[ "$code" -ne 0 && "$STOP_ON_ERROR" == 1 ]]; then
    exit "$code"
  fi
done

echo "Done. Summary: $SUMMARY_CSV"
