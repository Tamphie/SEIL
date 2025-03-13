import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quat):
    """
    Convert quaternion [x, y, z, w] into a 3x3 rotation matrix.
    
    :param quat: Array-like, shape (4,), with quaternion in [x, y, z, w] format.
    :return: A 3x3 numpy array (rotation matrix).
    """
    # Create a Rotation object from the quaternion
    r = R.from_quat(quat)  # [x, y, z, w]
    # Convert to a 3x3 rotation matrix
    rot_matrix = r.as_matrix()
    return rot_matrix

def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation representation back to a 3x3 rotation matrix.
    """
    # rot6d is shaped [6], e.g. [rx1, ry1, rz1, rx2, ry2, rz2]
    x1 = rot6d[0:3]
    x2 = rot6d[3:6]
    # Normalize
    x1_norm = x1 / np.linalg.norm(x1)
    # Make x2 orthonormal to x1
    x2_ortho = x2 - x1_norm * np.dot(x1_norm, x2)
    x2_norm = x2_ortho / np.linalg.norm(x2_ortho)
    x3 = np.cross(x1_norm, x2_norm)
    R = np.stack([x1_norm, x2_norm, x3], axis=-1)
    return R

def matrix_to_rot6d(R):
    """
    Convert 3x3 rotation matrix back to 6D representation.
    """
    # The 6D representation can be the first two columns of R
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)


def quaternion_to_6d(q):
    # Convert quaternion to rotation matrix
    r = R.from_quat(q)
    rot_matrix = r.as_matrix()  # 3x3 matrix

    # Take the first two columns
    m1 = rot_matrix[:, 0]
    m2 = rot_matrix[:, 1]

    # Flatten to 6D vector
    rot_6d = np.concatenate([m1, m2], axis=-1)
    return rot_6d
def get_contact_point_position(demo_dir, step_idx, contact_label):
    label_row = contact_label[step_idx]
    contact_indices = np.where(label_row == 1)[0]
    if len(contact_indices) == 0:
        raise ValueError(f"No contact found at step {step_idx} in episode 0!")
    min_index = contact_indices[0]
    points = np.load(f"{demo_dir}/episode_30/pcd_from_mesh/{step_idx}.npy")
    contact_point = points[min_index]
    return contact_point

def transform_pose_to_local(global_pose, contact_point, door_quat):
    pos = global_pose[:3]
    rot6d = global_pose[3:9]
    
    # Convert door quaternion to rotation matrix
    R_door = quaternion_to_matrix(door_quat)
    
    # Convert 6D rotation to a 3x3 matrix
    R_ee_global = rot6d_to_matrix(rot6d)
    
    # Compute local position:
    #   local_pos = R_door^T * (pos - contact_point)
    local_pos = R_door.T.dot(pos - contact_point)
    
    # Compute local orientation:
    #   local_ori = R_door^T * R_ee_global
    local_ori = R_door.T.dot(R_ee_global)
    
    # Convert back to 6D
    local_rot6d = matrix_to_rot6d(local_ori)
    
    # If you have a gripper dimension after the 9th element:
    if global_pose.shape[0] > 9:
        gripper_val = global_pose[9:]
        # print("Gripper value detected!")
        return np.concatenate([local_pos, local_rot6d, gripper_val], axis=0)
    else:
        return np.concatenate([local_pos, local_rot6d], axis=0)
            
def get_door_frame_poses(demo_dir):
    task_data = np.load(f"{demo_dir}/episode_30/task_data.npy", allow_pickle=True)
    door_poses = []
    for i, row in enumerate(task_data):
        row_list = row.tolist()
        object = 'door_main_visible'  # Adjust as per your data format
        try:
            idx = row_list.index(object)
            current_pose = row_list[idx-7:idx]  # Extract pose preceding the keyword
            position = np.array(current_pose[:3], dtype=np.float64)
            quaternion = np.array(current_pose[3:], dtype=np.float64)
            door_poses.append((position, quaternion))
        except ValueError:
            print(f"Object 'door_main_visible' not found in row {i}. Skipping...")
            door_poses.append(None)
    return door_poses

def process_episode(demo_dir, k=10):
    contact_label = np.load(f"{demo_dir}/episode_30/contact_label.npy").astype(np.float32)
    gripper_states = np.load(f"{demo_dir}/episode_30/gripper_states.npy").astype(np.float32)
    door_frame_poses = get_door_frame_poses(demo_dir)
    
    transformed_data = []
    
    for step_idx in range(len(gripper_states) - k + 1):
        if door_frame_poses[step_idx] is None:
            continue
        door_position, door_quat = door_frame_poses[step_idx]  # (pos, quat)
        contact_pos = get_contact_point_position(demo_dir, step_idx, contact_label)
        
        local_subseq = []
        for i in range(k):
            ee_pose = gripper_states[step_idx + i]
            ee_pos = ee_pose[:3]
            ee_quat = ee_pose[3:7]

            ee_rot_6d = np.array([quaternion_to_6d(ee_quat)]).reshape(-1)
            ee_pose = np.concatenate([ee_pos, ee_rot_6d,[ee_pose[-1]]], axis=-1)

            local_pose = transform_pose_to_local(ee_pose, contact_pos, door_quat)
            local_subseq.append(local_pose)
        
        transformed_data.append({
            'action': np.array(local_subseq, dtype=np.float32),
            'point': contact_pos,
            'pose': door_quat
        })
    
    output_path = f"{demo_dir}/episode_30/validate_training_input_obc.npy"
    np.save(output_path, transformed_data)
    print(f"Saved transformed data to {output_path}")


if __name__ == "__main__":
    demo_directory = "data/open_door"  # Set this to the correct path
    process_episode(demo_directory)
