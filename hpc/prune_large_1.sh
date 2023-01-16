#!/bin/bash

#SBATCH --job-name=prune-l-1
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
LOGDIR="${VSC_SCRATCH}/results-large"

# Prune the results.
python3 $VSC_HOME/distributional-dominance/prune_results.py \
--log-dir "$LOGDIR" \
--seed 1 \
--env large\
--prune dds cdds pf ch
