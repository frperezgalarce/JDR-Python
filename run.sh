#!/bin/bash
#SBATCH --job-name=jdr_dbscan
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=16G
#SBATCH --time=24:00:00

# -----------------------------------------------------------------------------
# NLHPC environment setup
# -----------------------------------------------------------------------------

echo "======================================================"
echo "Starting JDR DBSCAN experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "======================================================"

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# -----------------------------------------------------------------------------
# Create logs directory
# -----------------------------------------------------------------------------

mkdir -p logs

# -----------------------------------------------------------------------------
# Load modules
# -----------------------------------------------------------------------------

module purge

# Example Python module (adjust if needed)
module load python/3.10

# -----------------------------------------------------------------------------
# Activate virtual environment
# -----------------------------------------------------------------------------

# Uncomment and edit if you use a virtual environment
# source ~/venvs/jdr/bin/activate

# -----------------------------------------------------------------------------
# Print environment info
# -----------------------------------------------------------------------------

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installed packages:"
pip list | head

echo "Installing required libraries..."
pip install numpy pandas matplotlib scipy astropy scikit-learn random typing subprocess tempfile pathlib

# -----------------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------------

SCRIPT="notebooks/run.py"

echo "Running script: $SCRIPT"

time python $SCRIPT

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Final status
# -----------------------------------------------------------------------------

echo "======================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment finished successfully"
else
    echo "Experiment failed with exit code $EXIT_CODE"
fi

echo "End time: $(date)"

echo "======================================================"

exit $EXIT_CODE