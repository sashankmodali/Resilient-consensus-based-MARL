#!/bin/bash

python main.py --algo GenPBE --summary_dir simulation_results --n_episodes 10000
python main.py --algo saddle_point --summary_dir simulation_results --n_episodes 10000
python main.py --algo RPBCAC --summary_dir simulation_results --n_episodes 10000