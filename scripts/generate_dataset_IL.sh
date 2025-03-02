#!/bin/bash

echo "$(which python3)"
python3 rlbench/dataset_generator.py --save_path ./data --tasks open_door --processes 10 --episodes_per_task 40 --variations 1
