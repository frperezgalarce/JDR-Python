#!/bin/bash
#SBATCH --job-name=jdr_validation
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Install requirements in your environment before submission; no job-time installs.
# The logs directory must exist before sbatch opens the output file.
mkdir -p logs
"${JDR_PYTHON:-python3}" main.py "$@"
