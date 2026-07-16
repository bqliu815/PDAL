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

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR

# Keep one implementation of the 134-instance protocol. The delegated runner
# records solver-native runtime and the guarded-sigma audit fields consumed by
# experiments/qp_maros/collect_qp_maros.py.
exec bash "${PROJECT_DIR}/scripts/run_qp_table1_134.sbatch"
