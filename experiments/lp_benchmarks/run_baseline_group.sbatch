#!/bin/bash
#SBATCH -J lp_base
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00

set -uo pipefail

BASELINE_ROOT=${BASELINE_ROOT:?third-party source/build root is required}
REPRO_ROOT=${REPRO_ROOT:?repository root is required}
SOLVER=${SOLVER:?solver id is required}
CELL_MANIFEST=${CELL_MANIFEST:?cell manifest is required}
OUT_ROOT=${OUT_ROOT:?output root is required}
GROUP_SIZE=${GROUP_SIZE:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JULIA=${JULIA:-$(command -v julia 2>/dev/null || true)}
CUDA_HOME=${CUDA_HOME:-}

if ! command -v module >/dev/null 2>&1; then source /etc/profile >/dev/null 2>&1 || true; fi
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE" >/dev/null 2>&1 || true
  fi
fi

if [[ -z "$CUDA_HOME" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
fi
if [[ -z "$CUDA_HOME" ]]; then
  echo "ERROR: set CUDA_HOME or make nvcc available on PATH" >&2
  exit 2
fi

export JULIA_CUDA_USE_BINARYBUILDER=false
export ROOT="$BASELINE_ROOT"
export LD_LIBRARY_PATH="$BASELINE_ROOT/src/cuPDLPx-C-v0.2.9/build:$BASELINE_ROOT/src/cuPDLPx-C-v0.2.9/build/_deps/pslp-build:$BASELINE_ROOT/src/HPR-LP-C-v0.1.2/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUPDLPX_C_BIN=${CUPDLPX_C_BIN:-$BASELINE_ROOT/src/cuPDLPx-C-v0.2.9/build/cupdlpx}
CUPDLPC_BIN=${CUPDLPC_BIN:-$BASELINE_ROOT/src/cuPDLP-C-v0.4.1/build_gpu/bin/plc}
HPRLP_C_BIN=${HPRLP_C_BIN:-$BASELINE_ROOT/src/HPR-LP-C-v0.1.2/build/solve_mps_file}
CUPDLP_JL_PROJECT=${CUPDLP_JL_PROJECT:-$BASELINE_ROOT/src/cuPDLP-jl-master}
HPRLP_JL_PROJECT=${HPRLP_JL_PROJECT:-$BASELINE_ROOT/src/HPR-LP-v0.1.0}
PARSER="$REPRO_ROOT/experiments/lp_benchmarks/parse_outputs.py"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/solver_outputs"
summary="$OUT_ROOT/summary_${TASK_ID}.csv"
echo "solver,solver_id,dataset,split,instance,tolerance,time_limit,status,time_sec,iterations,objective,rel_gap,rel_primal,rel_dual,error" > "$summary"

start_row=$((TASK_ID * GROUP_SIZE + 2))
end_row=$((start_row + GROUP_SIZE - 1))
failed=0

while IFS=$'\t' read -r dataset split instance tolerance time_limit mps; do
  [[ -n "$instance" ]] || continue
  mps=${mps%$'\r'}
  if [[ ! -r "$mps" ]]; then
    echo "missing input: $mps" >&2
    failed=1
    continue
  fi
  instance_dir="$OUT_ROOT/solver_outputs/$instance"
  mkdir -p "$instance_dir"
  wall_limit=$(python3 -c "tl=float('$time_limit'); print(int(tl + max(600.0, min(1800.0, 0.10*tl))))")

  case "$SOLVER" in
    cupdlpx_c)
      "$CUPDLPX_C_BIN" "$mps" "$instance_dir" \
        --time_limit "$time_limit" --eps_opt "$tolerance" --eps_feas "$tolerance" \
        --no_presolve --quiet </dev/null > "$instance_dir/stdout.log" 2> "$instance_dir/stderr.log"
      code=$?
      python3 "$PARSER" cupdlpx_c "cuPDLPx(C)" "$dataset" "$split" "$tolerance" "$time_limit" "$instance" \
        "$instance_dir/${instance}_summary.txt" >> "$summary"
      ;;
    cupdlpc)
      "$CUPDLPC_BIN" -fname "$mps" -out "$instance_dir/summary.json" \
        -dTimeLim "$time_limit" -dPrimalTol "$tolerance" -dDualTol "$tolerance" \
        -dGapTol "$tolerance" -dFeasTol "$tolerance" -ifPre 0 \
        </dev/null > "$instance_dir/stdout.log" 2> "$instance_dir/stderr.log"
      code=$?
      python3 "$PARSER" cupdlpc "cuPDLP-C" "$dataset" "$split" "$tolerance" "$time_limit" "$instance" \
        "$instance_dir/summary.json" >> "$summary"
      ;;
    cupdlp_jl)
      if [[ -z "$JULIA" || ! -x "$JULIA" ]]; then
        echo "Julia executable not found; set JULIA explicitly." >&2
        exit 2
      fi
      timeout "$wall_limit" "$JULIA" --project="$CUPDLP_JL_PROJECT" \
        "$REPRO_ROOT/experiments/lp_benchmarks/run_cupdlp_jl_cli.jl" \
        --instance_path "$mps" --output_directory "$instance_dir" \
        --tolerance "$tolerance" --time_sec_limit "$time_limit" \
        </dev/null > "$instance_dir/stdout.log" 2> "$instance_dir/stderr.log"
      code=$?
      python3 "$PARSER" cupdlp_jl "cuPDLP.jl" "$dataset" "$split" "$tolerance" "$time_limit" "$instance" \
        "$instance_dir/${instance}_summary.json" >> "$summary"
      ;;
    hprlp_jl)
      if [[ -z "$JULIA" || ! -x "$JULIA" ]]; then
        echo "Julia executable not found; set JULIA explicitly." >&2
        exit 2
      fi
      timeout "$wall_limit" "$JULIA" --project="$HPRLP_JL_PROJECT" \
        "$REPRO_ROOT/experiments/lp_benchmarks/run_hprlp_one.jl" \
        "$mps" "$instance_dir/one.csv" "$tolerance" "$time_limit" \
        </dev/null > "$instance_dir/stdout.log" 2> "$instance_dir/stderr.log"
      code=$?
      python3 "$PARSER" hprlp_jl "HPR-LP.jl" "$dataset" "$split" "$tolerance" "$time_limit" "$instance" \
        "$instance_dir/one.csv" >> "$summary"
      ;;
    hprlp_c)
      timeout "$wall_limit" "$HPRLP_C_BIN" -i "$mps" --tol "$tolerance" \
        --time-limit "$time_limit" --presolve false \
        </dev/null > "$instance_dir/stdout.log" 2> "$instance_dir/stderr.log"
      code=$?
      python3 "$PARSER" hprlp_c "HPR-LP-C" "$dataset" "$split" "$tolerance" "$time_limit" "$instance" \
        "$instance_dir/stdout.log" >> "$summary"
      ;;
    *)
      echo "unknown solver: $SOLVER" >&2
      exit 2
      ;;
  esac
  if [[ $code -ne 0 ]]; then failed=1; fi
done < <(sed -n "${start_row},${end_row}p" "$CELL_MANIFEST")

exit "$failed"
