#!/bin/bash

#SBATCH --job-name=large-momdp
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
pip install --user gym
pip install --user mo-gym
pip install --user ramo
pip install --user pymoo
pip install --user scikit-learn
pip install --user POT
pip install --user pulp

# Define the log directory.
LOGDIR="${VSC_SCRATCH}/results"

# Run the experiments.
python3 $VSC_HOME/distributional-dominance/experiments.py \
--log-dir "$LOGDIR" \
--seed 1 2 3 4 5 \
--env large \
--warmup 50000 \
--num-episodes 2000 \
--save \
--log-every 5000 \
--num-threads 1