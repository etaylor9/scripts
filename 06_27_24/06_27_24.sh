#!/bin/bash
#SBATCH --job-name=hexagonalLatticenoProbePoints
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
outdir=$GROUP_SCRATCH/simulations/results/06_27_24/sym/$SLURM_JOB_ID
mkdir -p $outdir

# Specify the Python script path
pyscript=$HOME/scripts/06_27_24/06_27_24.py

# Copy the Python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

# Define the values of f as fractions
f_values=("1/3" "2/3" "1")

for f in "${f_values[@]}"; do
    # Evaluate the fraction using awk
    f_decimal=$(awk "BEGIN {print $f}")

    # Run the Python script with the --f and --outdir arguments
    python $pyscript --f $f_decimal --outdir $outdir
done


# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir

