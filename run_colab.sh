#!/bin/bash

python main.py --algo GenPBE --summary_dir simulation_results --n_episodes 10000 --fast_lr 0.02 --slow_lr 0.01
python main.py --algo saddle_point --summary_dir simulation_results --n_episodes 10000 --fast_lr 0.02 --slow_lr 0.01
python main.py --algo RPBCAC --summary_dir simulation_results --n_episodes 10000 --fast_lr 0.02 --slow_lr 0.01