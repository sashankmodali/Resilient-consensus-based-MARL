#!/bin/bash
variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate deep-learning
rm -rf simulation_results/tensorboard/GenPBE
python main.py --algo GenPBE --summary_dir simulation_results --n_episodes 10000 --fast_lr 0.02 --slow_lr 0.01