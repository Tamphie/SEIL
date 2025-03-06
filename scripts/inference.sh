#!/bin/bash

# Set variables
task_name="open_door"
episode_length=180
policy_class="ACT"  # ["ACT", "Diffusion"]
visual_encoder="resnet18"  # ["dinov2", "resnet18"]
variant="vits14"  # ["vits14", "vitb14", "vitl14", "vitg14"]
predict_value="ee_pos_ori" # ["joint_states", "ee_pos_ori", "ee_delta_pos_ori", "ee_relative_pos_ori"]
if [ "$predict_value" = "joint_states" ]; then
    state_dim=8
else
    state_dim=10
fi
obs_type="rgbd"
# Export environment variables
export MASTER_ADDR='localhost'  # Use the appropriate master node address
export MASTER_PORT=12345        # Use any free port

#  --temporal_agg \
# Run the Python script
python SEIL_infer.py \
    --ckpt_dir check_point\
    --ckpt_name policy_best.ckpt \
    --task_name ${task_name} \
    --policy_class ${policy_class} \
    --visual_encoder ${visual_encoder} \
    --variant ${variant} \
    --seed 0 \
    --state_dim "$state_dim" \
    --predict_value ${predict_value} \
    --obs_type ${obs_type} \
    --episode_len ${episode_length}  \
