#!/bin/bash
#SBATCH --job-name=NbN
#SBATCH --output=slurm-%A.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00

set -x

module load media ffmpeg

# Activate conda environment
source $GROUP_HOME/miniconda3/bin/activate
conda activate TDGL

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Create output directory in scratch space
outdir=$GROUP_SCRATCH/simulations/results/07_15_24/NbN/$SLURM_JOB_ID
mkdir -p $outdir

# Specify the Python script path
pyscript=$HOME/scripts/07_15_24/07_15_24_NbN.py

# Copy the Python script and this shell script to the results directory
cp -u $pyscript $outdir/ || { echo "Failed to copy Python script"; exit 1; }
cp -u $0 $outdir/ || { echo "Failed to copy shell script"; exit 1; }

# Define the value of f
f="1/4"

coherence_length="30"

# Evaluate the fraction using awk
f_decimal=$(awk "BEGIN {print $f}")

# Run the Python script with the --f and --outdir arguments
python $pyscript --f $f_decimal --coherence_length $coherence_length --outdir $outdir || { echo "Python script failed for coherence_length $coherence_length"; exit 1; }

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir || { echo "Failed to move slurm output log"; exit 1; }