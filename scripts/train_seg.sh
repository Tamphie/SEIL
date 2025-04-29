#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set variables
task_name="open_door"
predict_value="ee_pos_ori" # ["joint_states", "ee_pos_ori"]
obs_type="pcd" # ["rgbd", "pcd"] 
policy_class="ACT"  # ["ACT", "Diffusion"]
visual_encoder="pointnet"  # ["dinov2", "resnet18", "pointnet"
variant="vits14"  # ["vits14", "vitb14", "vitl14", "vitg14"]
# Export environment variables
# export MASTER_ADDR='localhost'  # Use the appropriate master node address
# export MASTER_PORT=12346        # Use any free port
# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# export MASTER_ADDR="localhost"
# Run the Python script
python train_seg.py \
    --task_name ${task_name} \
    --batch_size 4 \
    --num_epochs 100 \
    --ckpt_dir check_point/${task_name}_${policy_class}_${visual_encoder}_${variant} \
    --seed 0 \
    --predict_value ${predict_value} \
    --obs_type ${obs_type} \
    --pointnet_dir pointnet2 \
    --seg_lr 0.001 \
    --seg_num_point 1024 \
    --seg_num_classes 2 \
    --seg_optimizer adam
    #--joint_training \
