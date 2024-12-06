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

def extract_door_frame_pose(file_path):
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
        

# Example usage
file_path = "data/open_door/episode_1/task_data.npy"  # Replace with the correct file path
extract_door_frame_pose(file_path)
