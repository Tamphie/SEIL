#!/bin/bash

# Set variables
task_name="open_door"
episode_length=180
policy_class="ACT"  # ["ACT", "Diffusion"]
visual_encoder="pointnet"  # ["dinov2", "resnet18", "pointnet"]
variant="vits14"  # ["vits14", "vitb14", "vitl14", "vitg14"]
predict_value="ee_pos_ori" # ["joint_states", "ee_pos_ori", "ee_delta_pos_ori", "ee_relative_pos_ori"]
if [ "$predict_value" = "joint_states" ]; then
    state_dim=8
else
    state_dim=10
fi
obs_type="pcd"
# Export environment variables
export MASTER_ADDR='localhost'  # Use the appropriate master node address
export MASTER_PORT=12345        # Use any free port
seg_checkpoint="check_point/seg_model/seg_model_best.ckpt"
# seg_checkpoint="check_point/${task_name}_${policy_class}_${visual_encoder}_${variant}/seg_model/seg_model_best.ckpt"
ckpt_dir="check_point/${task_name}_${policy_class}_${visual_encoder}_${variant}"
    # 
# Run the Python script
python SEIL_infer.py \
    --ckpt_dir ${ckpt_dir} \
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
    --chunk_size 50 \
    --temporal_agg \
    --use_segmentation \
    --seg_checkpoint ${seg_checkpoint} \
    --pointnet_dir pointnet2 \
    --seg_threshold 0.7
