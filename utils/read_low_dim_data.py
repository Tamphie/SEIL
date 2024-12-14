#This is to read the specific object's state in low_dim_data of a task episode:
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quaternion):
    """
    Convert quaternion (qx, qy, qz, qw) to Euler angles (yaw, pitch, roll).
    :param quaternion: List of quaternion values [qx, qy, qz, qw]
    :return: Yaw, Pitch, Roll in degrees
    """
    rotation = R.from_quat(quaternion)
    yaw, pitch, roll = rotation.as_euler('zyx', degrees=True)  # ZYX convention (yaw, pitch, roll)
    return yaw, pitch, roll

def extract_door_frame_pose():
    file_path = "data/open_door/episode_1/task_data.npy"
    """
    Extract and print the pose of the door frame for each row in the .npy file.

    :param file_path: Path to the task_data.npy file
    """
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)  # Use allow_pickle if the data contains mixed types
    previous_pose = None

    for i, row in enumerate(data):
        row_list = row.tolist()  # Convert row to a Python list (if needed)
        # Find the 'door_frame' keyword in the row
        object = 'door_main_visible'
        idx = row_list.index(object)
        # Extract the pose elements preceding 'door_frame'
        # current_pose = row_list[idx:idx+2]
        current_pose = row_list[idx-7:idx]
        current_orientation = current_pose[-4:]
        if current_pose != previous_pose:
            yaw, pitch, roll = quaternion_to_euler(current_orientation)
            print(f"Line {i}: {object} orientation (yaw, pitch, roll): ({yaw:.2f}, {pitch:.2f}, {roll:.2f})")
            # print(f"Line {i}: {object} pose: {current_pose}")
            previous_pose = current_pose
        


def extract_contact_spoint():


    # File paths
    pcd_base_path = "data/open_door/episode_0/pcd_from_mesh"
    dist_data_path = "data/open_door/episode_0/dist_data.npy"

    # Load dist_data
    dist_data = np.load(dist_data_path)

    # Determine the reference point from dist_data
    # Find the row with the smallest distance (last element of each row)
    min_distance_index = np.argmin(dist_data[:, -1])
    reference_point = dist_data[min_distance_index, :3]  # Extract first 3 elements
    print(f"min_dist:{min_distance_index},dist: {dist_data[min_distance_index, -1] } point: {reference_point}")

    # Initialize output
    results = []
    tolerance = 1e-4
    # Iterate through steps to compare with the consistent reference point
    for step in range(len(dist_data)):  # Assuming one step per pcd_from_mesh file
        # Load pcd_from_mesh for the current step
        pcd_file = f"{pcd_base_path}/{step}.npy"
        pcd_from_mesh = np.load(pcd_file)

        # Check if the reference point exists in the current point cloud
        match_indices = np.where(np.all(np.isclose(pcd_from_mesh, reference_point, atol=tolerance), axis=1))[0]
        if match_indices.size > 0:
            results.append((step, match_indices.tolist()))  # Store step index and point index
        else:
            results.append((step, None))  # No match found

    # Output results
    for step, result in results:
        print(f"Step {step}: Point Index in PCD = {result}")

def print_dis():


    # File paths
    pcd_base_path = "data/open_door/episode_0/pcd_from_mesh"
    dist_data_path = "data/open_door/episode_0/dist_data.npy"

    # Load dist_data
    dist_data = np.load(dist_data_path)
    for step in range(len(dist_data)): 
        print(f"Step {step}: {dist_data[step, -1]}")

print_dis()