#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PROJECT_DIR="${TEST_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASELINE_PROJECT_DIR="${BASELINE_PROJECT_DIR:-}"
SBATCH_RUN="${SBATCH_RUN:-$SCRIPT_DIR/run_qp_table1_134.sbatch}"

DATA_DIR="${DATA_DIR:-$TEST_PROJECT_DIR/../Maros-Meszaros-Table1-134}"
OUT_ROOT="${OUT_ROOT:-$TEST_PROJECT_DIR/validation_runs_table1_134}"
TIME_LIM="${TIME_LIM:-1000}"
ITER_LIM="${ITER_LIM:-2147483647}"
EPS_OPT="${EPS_OPT:-1e-6}"
EPS_FEAS="${EPS_FEAS:-1e-6}"
EVAL_FREQ="${EVAL_FREQ:-200}"
VERBOSE="${VERBOSE:-0}"
MAX_CASES="${MAX_CASES:-0}"
INSTANCE_LIST="${INSTANCE_LIST:-$DATA_DIR/instances_table1_134.txt}"
FA_CP_SIGMA="${FA_CP_SIGMA:-0}"
FA_CP_SIGMA_UPDATE="${FA_CP_SIGMA_UPDATE:-guarded}"
BASE_RUN_TAG="${BASE_RUN_TAG:-baseline}"
TEST_RUN_TAG="${TEST_RUN_TAG:-fa_cp}"

mkdir -p "$OUT_ROOT"

test_export="ALL,PROJECT_DIR=$TEST_PROJECT_DIR,RUN_NAME=fa_cp,DATA_DIR=$DATA_DIR,OUT_ROOT=$OUT_ROOT,RUN_TAG=$TEST_RUN_TAG,TIME_LIM=$TIME_LIM,ITER_LIM=$ITER_LIM,EPS_OPT=$EPS_OPT,EPS_FEAS=$EPS_FEAS,EVAL_FREQ=$EVAL_FREQ,VERBOSE=$VERBOSE,MAX_CASES=$MAX_CASES,FA_CP_USE_AUGMENTATION=1,FA_CP_SIGMA=$FA_CP_SIGMA,FA_CP_SIGMA_UPDATE=$FA_CP_SIGMA_UPDATE"

if [[ -n "$INSTANCE_LIST" ]]; then
  test_export="${test_export},INSTANCE_LIST=$INSTANCE_LIST"
fi

test_job="$(sbatch --parsable --export="$test_export" "$SBATCH_RUN")"

echo "Submitted FA_CP job: $test_job"

if [[ -n "$BASELINE_PROJECT_DIR" ]]; then
  # Optional baseline comparison. Set BASELINE_PROJECT_DIR to a compatible
  # non-augmented QP baseline checkout if you want to run it side by side.
  base_export="ALL,PROJECT_DIR=$BASELINE_PROJECT_DIR,RUN_NAME=baseline_fa_cp_qp,DATA_DIR=$DATA_DIR,OUT_ROOT=$OUT_ROOT,RUN_TAG=$BASE_RUN_TAG,TIME_LIM=$TIME_LIM,ITER_LIM=$ITER_LIM,EPS_OPT=$EPS_OPT,EPS_FEAS=$EPS_FEAS,EVAL_FREQ=$EVAL_FREQ,VERBOSE=$VERBOSE,MAX_CASES=$MAX_CASES,FA_CP_USE_AUGMENTATION=0"
  if [[ -n "$INSTANCE_LIST" ]]; then
    base_export="${base_export},INSTANCE_LIST=$INSTANCE_LIST"
  fi
  base_job="$(sbatch --parsable --export="$base_export" "$SBATCH_RUN")"
  echo "Submitted optional baseline job: $base_job"
fi
echo
echo "Output root: $OUT_ROOT"
echo "Test summary:     $OUT_ROOT/$TEST_RUN_TAG/summary.csv"
if [[ -n "${base_job:-}" ]]; then
  echo "Baseline summary: $OUT_ROOT/$BASE_RUN_TAG/summary.csv"
fi
