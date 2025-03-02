import os
import argparse
import numpy as np
import zarr
from PIL import Image
from copy import deepcopy
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description="Process RLBench data into Zarr format.")
parser.add_argument("--pcd", type=str, choices=["pcd_from_mesh", "front_pcd", "mesh_pcd_label"], default="pcd_from_mesh", help="Point cloud data type")
parser.add_argument("--state", type=str, choices=["joint_positions", "gripper_states"], default="joint_positions", help="State representation")
parser.add_argument("--action", type=str, choices=["joint_positions", "gripper_states"], default="joint_positions", help="Action representation")
args = parser.parse_args()

# ✅ Update these paths
rlbench_data_path = "data/open_door"  # RLBench dataset path
output_zarr_path = os.path.join(rlbench_data_path, "rlbench_open_door_expert.zarr")  # Output Zarr file
# output_zarr_path = os.path.join(rlbench_data_path, "drill_40demo_1024.zarr")
unzip_data_path = "data/zarr1/unzip"  # Path to unzip the RLBench dataset

def save_rlbench_to_zarr(rlbench_data_path, output_zarr_path, args):
    """Convert RLBench dataset into DP3-compatible Zarr format."""
    
    # ✅ Initialize storage lists for DP3 format
    img_arrays = []
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    total_count = 0

    # ✅ Loop over each episode directory
    episode_dirs = sorted(os.listdir(rlbench_data_path))  # List all episodes
    for episode_idx, episode_name in enumerate(tqdm(episode_dirs, desc="Processing Episodes")):
        # Process only folders matching "episode_n" format
        if not re.match(r"^episode_\d+$", episode_name):
            continue  
        
        episode_path = os.path.join(rlbench_data_path, episode_name)
        if not os.path.isdir(episode_path):
            continue  # Skip if not a directory

        # ✅ Load full-episode data
        gripper_states = np.load(os.path.join(episode_path, "gripper_states.npy"))  # (num_steps, D_action)
        joint_positions = np.load(os.path.join(episode_path, "joint_positions.npy"))  # (num_steps, D_state)

        num_steps = gripper_states.shape[0]  # Total steps in this episode

        img_arrays_sub = []
        point_cloud_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []

        for step_idx in range(num_steps):
            # ✅ Load point cloud
            pcd_path = os.path.join(episode_path, args.pcd, f"{step_idx}.npy")
            point_cloud = np.load(pcd_path) if os.path.exists(pcd_path) else np.zeros((512, 3))
            # if args.pcd == "front_pcd" or args.pcd == "mesh_pcd_label":
            #     point_cloud = point_cloud.reshape(point_cloud.shape[0], -1, point_cloud.shape[-1])
            #     print(f"Reshaping: point cloud shape: {point_cloud.shape}")

            # ✅ Load RGB image
            img_path = os.path.join(episode_path, "front_rgb", f"{step_idx}.png")
            img = np.array(Image.open(img_path)) if os.path.exists(img_path) else np.zeros((84, 84, 3), dtype=np.uint8)

            # ✅ Append data
            img_arrays_sub.append(img)
            point_cloud_arrays_sub.append(point_cloud)
            
            state_arrays_sub.append(joint_positions[step_idx] if args.state == "joint_positions" else gripper_states[step_idx])
            action_arrays_sub.append(joint_positions[step_idx] if args.action == "joint_positions" else gripper_states[step_idx])


        # ✅ Store episode information
        total_count += num_steps
        episode_ends_arrays.append(deepcopy(total_count))

        img_arrays.extend(deepcopy(img_arrays_sub))
        point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
        state_arrays.extend(deepcopy(state_arrays_sub))
        action_arrays.extend(deepcopy(action_arrays_sub))

        print(f"Processed {episode_name}: {num_steps} steps")

    # ✅ Convert lists to NumPy arrays
    img_arrays = np.stack(img_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    if args.pcd == "front_pcd" or args.pcd == "mesh_pcd_label":
        point_cloud_arrays = point_cloud_arrays.reshape(point_cloud_arrays.shape[0], -1, point_cloud_arrays.shape[-1])
        print(f"Reshaping: point cloud shape: {point_cloud_arrays.shape}")

    state_arrays = np.stack(state_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # ✅ Create Zarr dataset
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_root = zarr.group(output_zarr_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # ✅ Save as Zarr
    zarr_data.create_dataset("img", data=img_arrays, dtype="uint8", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("state", data=state_arrays, dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("point_cloud", data=point_cloud_arrays, dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=action_arrays, dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arrays, dtype="int64", overwrite=True, compressor=compressor)

    print(f"✅ Successfully saved RLBench data in DP3 Zarr format at {output_zarr_path}")
    
    # ✅ Create a text file to store argument setup
    args_txt_path = os.path.join(output_zarr_path, "args_setup.txt")
    with open(args_txt_path, "w") as f:
        f.write(f"pcd: {args.pcd}\n")
        f.write(f"state: {args.state}\n")
        f.write(f"action: {args.action}\n")

    print(f"✅ Saved argument setup to {args_txt_path}")


def read_zarr_data(zarr_path, output_path):
    """Read Zarr dataset and save it back as .npy files."""
    print(f"📂 Reading Zarr file from: {zarr_path}")

    # ✅ Load Zarr dataset
    store = zarr.open(zarr_path, mode='r')
    for key in store.keys():
        print(f"🔑 Key: {key}")
    valid_keys = [key for key in store["data"].keys()]
    print("✅ Available keys in Zarr file:", valid_keys)
   
    os.makedirs(output_path, exist_ok=True)
    # ✅ Convert to NumPy and save
    for key in valid_keys:
        array = store[f"data/{key}"][...]  # Read NumPy array
        print(f"Saving {key} with shape {array.shape}")
        output_file = os.path.join(output_path, f"{key}.npy")
        np.save(output_file, array)  # Save as .npy

    print("✅ Zarr extraction completed successfully!")


if __name__ == "__main__":
    # ✅ Step 1: Convert RLBench data to Zarr
    save_rlbench_to_zarr(rlbench_data_path, output_zarr_path, args)
    # ✅ Step 2: Read and save back as .npy for verification
    read_zarr_data(output_zarr_path, unzip_data_path)
