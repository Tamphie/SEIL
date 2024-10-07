#!/bin/bash

# Set variables
task_name="open_box"
policy_class="ACT"  # ["ACT", "Diffusion"]
visual_encoder="resnet18"  # ["dinov2", "resnet18"]
variant="vits14"  # ["vits14", "vitb14", "vitl14", "vitg14"]
predict_value="ee_pos_ori" # ["joint_states", "ee_pos_ori"]

# Conditional chunk_size setting
if [ "$policy_class" == "ACT" ]; then
    chunk_size=100
elif [ "$policy_class" == "Diffusion" ]; then
    chunk_size=16
else
    echo "Invalid policy_class: $policy_class"
    exit 1
fi

# Export environment variables
export MASTER_ADDR='localhost'  # Use the appropriate master node address
export MASTER_PORT=12345        # Use any free port

# Run the Python script
python SEIL_train.py \
    --policy_class ${policy_class} \
    --task_name ${task_name} \
    --batch_size 32 \
    --chunk_size ${chunk_size} \
    --num_epochs 300 \
    --ckpt_dir check_point/${task_name}_${policy_class}_${visual_encoder}_${variant} \
    --seed 0 \
    --predict_value ${predict_value} \
    --visual_encoder ${visual_encoder} \
    --variant ${variant}
