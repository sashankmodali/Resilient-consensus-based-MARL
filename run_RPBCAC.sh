#!/bin/bash
variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate deep-learning
python main.py --algo RPBCAC --summary_dir simulation_results --n_episodes 10000 