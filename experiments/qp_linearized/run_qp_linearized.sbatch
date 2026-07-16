#!/usr/bin/env bash
#SBATCH -J rhr_qp1000
#SBATCH -p NV_H100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-199%24
#SBATCH -o rhr_qp1000-%A_%a.out
#SBATCH -e rhr_qp1000-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the generated 1000-instance directory}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/paper_runs/qp_linearized_1000}"
PROFILES="${PROFILES:-$SCRIPT_DIR/profiles.psv}"
NCHUNKS="${NCHUNKS:-40}"
EXPECTED_INSTANCES="${EXPECTED_INSTANCES:-1000}"

profile_count=$(awk -F '|' 'NR > 1 && NF {count++} END {print count + 0}' "$PROFILES")
task_id="${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"
profile_index=$((task_id / NCHUNKS + 1))
chunk_index=$((task_id % NCHUNKS))
if ((profile_index < 1 || profile_index > profile_count)); then
  echo "ERROR: task $task_id maps to invalid profile $profile_index" >&2
  exit 2
fi

profile_line=$(awk -F '|' -v row="$((profile_index + 1))" 'NR == row {print}' "$PROFILES")
IFS='|' read -r profile source_dir binary_name binary_env eval_freq extra_args <<< "$profile_line"
if [[ -z "$profile" || -z "$binary_name" || -z "$eval_freq" ]]; then
  echo "ERROR: unable to read profile $profile_index from $PROFILES" >&2
  exit 2
fi

all_instances="$DATASET_ROOT/lists/all_instances.txt"
if [[ ! -f "$all_instances" ]]; then
  echo "ERROR: generated instance list not found: $all_instances" >&2
  exit 2
fi
instance_count=$(grep -cve '^[[:space:]]*$' "$all_instances")
if [[ "$instance_count" -ne "$EXPECTED_INSTANCES" ]]; then
  echo "ERROR: expected $EXPECTED_INSTANCES instances, found $instance_count" >&2
  exit 2
fi
if ((EXPECTED_INSTANCES % NCHUNKS != 0)); then
  echo "ERROR: EXPECTED_INSTANCES must be divisible by NCHUNKS" >&2
  exit 2
fi

if [[ -n "$binary_env" ]]; then
  binary="${!binary_env:-}"
  if [[ -z "$binary" ]]; then
    echo "ERROR: set $binary_env to the external comparison executable" >&2
    exit 2
  fi
else
  if [[ -z "$source_dir" ]]; then
    echo "ERROR: profile $profile has neither a source directory nor a binary environment variable" >&2
    exit 2
  fi
  project_dir="$REPO_ROOT/$source_dir"
  binary="$project_dir/build/$binary_name"
  if [[ ! -x "$binary" && -x "$project_dir/build/bin/$binary_name" ]]; then
    binary="$project_dir/build/bin/$binary_name"
  fi
  if [[ ! -x "$binary" ]]; then
    echo "ERROR: build the selected solver first: $project_dir/build/$binary_name" >&2
    exit 2
  fi
fi
if [[ ! -x "$binary" ]]; then
  echo "ERROR: executable not found: $binary" >&2
  exit 2
fi

chunk_tag=$(printf '%02d' "$chunk_index")
chunk_dir="$OUT_ROOT/manifests/$profile"
chunk_file="$chunk_dir/chunk_$chunk_tag.txt"
mkdir -p "$chunk_dir"
awk -v chunk="$chunk_index" -v count="$NCHUNKS" \
  'NF && (seen++ % count == chunk) {print}' "$all_instances" > "$chunk_file"

expected_chunk=$((EXPECTED_INSTANCES / NCHUNKS))
actual_chunk=$(wc -l < "$chunk_file" | tr -d ' ')
if [[ "$actual_chunk" -ne "$expected_chunk" ]]; then
  echo "ERROR: chunk $chunk_index contains $actual_chunk rather than $expected_chunk instances" >&2
  exit 2
fi

run_out="$OUT_ROOT/results/$profile/chunk_$chunk_tag"
printf 'profile=%s\nsource_dir=%s\nbinary=%s\nchunk=%s\n' \
  "$profile" "${source_dir:-external}" "$binary" "$chunk_file"
env \
  "BIN=$binary" \
  "INSTANCE_LIST=$chunk_file" \
  "OUT_ROOT=$run_out" \
  "RUN_NAME=$profile" \
  "TIME_LIM=60" \
  "EPS_OPT=1e-5" \
  "EPS_FEAS=1e-5" \
  "EVAL_FREQ=$eval_freq" \
  "EXTRA_ARGS=$extra_args" \
  bash "$SCRIPT_DIR/run_profile_chunk.sh"
