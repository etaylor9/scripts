#!/bin/bash
#SBATCH --job-name=hexagonalLattice
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
##### !!!!!!!!!!!!!!!! CHANGE THIS !!!!!!!!!!!!!!!! #####
outdir=$GROUP_SCRATCH/simulations/results/06_12_24/$SLURM_JOB_ID
mkdir -p $outdir

##### !!!!!!!!!!!!!!!! CHANGE THIS !!!!!!!!!!!!!!!! #####
pyscript=$HOME/scripts/06_12_24/06_12_24_.py 


# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --directory=$outdir

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir