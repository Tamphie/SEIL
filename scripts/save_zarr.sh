#!/bin/bash
# bash scripts/save_zarr.sh front_pcd gripper_states joint_positions

python3 utils/save_zarr.py --pcd "$1" --state "$2" --action "$3"
