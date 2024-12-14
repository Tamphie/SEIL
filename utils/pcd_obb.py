import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
import time
def extract_door_frame_pose(task_data_path):
    """
    Extract the door frame pose for each step from task_data.npy.

    :param task_data_path: Path to the task_data.npy file.
    :return: List of poses, where each pose is a tuple (position, quaternion).
    """
    data = np.load(task_data_path, allow_pickle=True)
    poses = []

    for i, row in enumerate(data):
        row_list = row.tolist()
        object = 'door_main_visible'  # Adjust as per your data format
        try:
            idx = row_list.index(object)
            current_pose = row_list[idx-7:idx]  # Extract pose preceding the keyword
            position = np.array(current_pose[:3], dtype=np.float64)
            quaternion = np.array(current_pose[3:], dtype=np.float64)
            poses.append((position, quaternion))
        except ValueError:
            print(f"Object 'door_main_visible' not found in row {i}. Skipping...")
            poses.append(None)

    return poses

def visualize_pcd_with_frame(pcd_path, task_data_path):
    """
    Visualize the point cloud and frame orientation for each step.

    :param pcd_path: Path to the directory containing per-step pcd_from_mesh files.
    :param task_data_path: Path to the task_data.npy file.
    """
    # Extract the door frame poses
    poses = extract_door_frame_pose(task_data_path)

    # Iterate through pcd files and corresponding poses
    for step in range(100,150):
        # Load the point cloud for the current step
        pcd_file = f"{pcd_path}/{step}.npy"
        pcd = np.load(pcd_file)

        # Get the corresponding pose
        if poses[step] is None:
            print(f"Skipping step {step} due to missing pose.")
            step += 1
            continue

        position, quaternion = poses[step]

        # Convert quaternion to rotation matrix
        rotation = R.from_quat(quaternion).as_matrix()

        # Plot the point cloud and frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', s=1, label='Point Cloud')

        # Draw the frame
        origin = position
        scale = 0.1  # Scale for the frame axes
        x_axis = origin + scale * rotation[:, 0]
        y_axis = origin + scale * rotation[:, 1]
        z_axis = origin + scale * rotation[:, 2]

        ax.quiver(*origin, *(x_axis-origin), color='r', label='X-axis')
        ax.quiver(*origin, *(y_axis-origin), color='g', label='Y-axis')
        ax.quiver(*origin, *(z_axis-origin), color='b', label='Z-axis')

        # Set plot limits and labels
        ax.set_xlim(origin[0]-0.5, origin[0]+0.5)
        ax.set_ylim(origin[1]-0.5, origin[1]+0.5)
        ax.set_zlim(origin[2]-0.5, origin[2]+0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f"Step {step}")
        plt.legend()
        plt.show()

            
def visualize_pcd(pcd_path):
    pcd_file = f"{pcd_path}/{0}.npy"
    points = np.load(pcd_file)
    points = points.reshape(-1, 3)
    print(f"shape:{points.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_pcd_as_vedio(pcd_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    for step in range(150):
        pcd_file = f"{pcd_path}/{step}.npy"
        points = np.load(pcd_file)
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.5)
    vis.destroy_window()


if __name__ == "__main__":
    # Set paths to the pcd_from_mesh files and task_data.npy
    pcd_path = "data/open_door/episode_0/pcd_from_mesh"  # Update with the correct path
    task_data_path = "data/open_door/episode_0/task_data.npy"  # Update with the correct path

    # Call the visualization function
    visualize_pcd(pcd_path)
