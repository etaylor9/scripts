#!/bin/bash
#SBATCH --job-name=hexagonalLatticenoProbePoints
#SBATCH --output=slurm-%A.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00

set -x

module load media ffmpeg

# GROUP_HOME="."

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate TDGL

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# GROUP_SCRATCH="."
outdir=$GROUP_SCRATCH/simulations/results/06_18_24/noProbe/$SLURM_JOB_ID
mkdir -p $outdir

pyscript=$HOME/scripts/06_18_24/06_18_24_noProbe.py #HAVE TO ACTUALLY PUT THE SCRIPT NAME HERE

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

# Define the values of f as fractions
f_values=("1/3" "2/3" "1")

for f in "${f_values[@]}"; do
    # Evaluate the fraction using awk
    f_decimal=$(awk "BEGIN {print $f}")

    # Run the Python script with the --f argument
    python $pyscript --f $f_decimal --directory=$outdir
done

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir