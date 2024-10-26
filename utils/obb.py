import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
def statistical_outlier(door_pcd):
    door_pcd, ind = door_pcd.remove_statistical_outlier(nb_neighbors=130, std_ratio=0.30)
    # each point will need more neighboring points to be considered an inlier (noclean:increase)
    # treshold of distance from nearby points of outlier to be removed (noclean:decrease)
    # Perform plane segmentation to isolate the flat door surface
    plane_model, inliers = door_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = door_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1, 0])  # Color the inlier points green for visualization
    return inlier_cloud

def get_cluster_color(labels,colors):
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1 labels)
    for label in unique_labels:
        color = colors[labels == label][0]  # Get the color for the current label
        print(f"Label {label}: Color {color[:3]}")  # Print only RGB channels

def iterate():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    directory = "data/open_door/episode_0/front_pcd"
    save_path = os.path.join(os.path.dirname(directory), "frame.npy")
    frames = []
    for filename in sorted(os.listdir(directory)):
        
        file_path = os.path.join(directory, filename)
        
        # Get the point cloud and OBB for the current file
        pcd, obb = obb_from_pcd(file_path)
        new_obb,R = adjust_obb(obb)
        frames.append(R)
        axes = create_axes(new_obb)
        # o3d.visualization.draw_geometries([ pcd,*axes])
    #     vis.add_geometry(pcd)
    #     vis.update_geometry(pcd)
    #     # Update the point cloud geometry with new points and colors
    #     if obb:
    #         vis.add_geometry(obb)
    #         vis.update_geometry(obb)
            
    #     # Refresh visualization
    #     vis.poll_events()
    #     vis.update_renderer()
    #     time.sleep(0.5)

    frames = np.array(frames)
    np.save(save_path, frames)
    # vis.destroy_window()

def adjust_obb(obb):
    # Extract the current rotation matrix from OBB
    R_original = obb.R

    # Step 1: Rotate 90 degrees around the y-axis to map x -> z, z -> -x
    rotation_matrix_y_90 = np.array([
        [0, 0, 1],  # x-axis maps to z-axis
        [0, 1, 0],  # y-axis remains y-axis
        [-1, 0, 0]  # z-axis maps to -x-axis
    ])

    # Step 2: Rotate 180 degrees around the new x-axis to map y -> -y, z -> -z
    rotation_matrix_x_180 = np.array([
        [-1, 0, 0],  # x-axis remains x-axis
        [0, -1, 0], # y-axis flips to -y
        [0, 0, 1]  # z-axis flips to -z
    ])

    # Apply the rotations sequentially to obtain the adjusted rotation matrix
    # R_adjusted = R_original @ rotation_matrix_y_90 @ rotation_matrix_x_180
    R_adjusted = R_original@ rotation_matrix_y_90 @ rotation_matrix_x_180
    # Update the OBB rotation matrix with the adjusted rotation
    obb.R = R_adjusted

    return obb, obb.R

def create_axes(obb):
    # Extract the OBB center and rotation matrix
    center = obb.center
    R = obb.R
    extents = obb.extent / 2  # Half-lengths along each axis

    # Define the axes based on OBB's rotation matrix and extent
    x_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 0] * extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    x_axis.colors = o3d.utility.Vector3dVector([(1, 0, 0)])  # Red for x-axis

    y_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 1] * extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    y_axis.colors = o3d.utility.Vector3dVector([(0, 1, 0)])  # Green for y-axis

    z_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 2] * extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    z_axis.colors = o3d.utility.Vector3dVector([(0, 0, 1)])  # Blue for z-axis

    return [x_axis, y_axis, z_axis]


def obb_from_pcd(file_path):
    
    points = np.load(file_path)
    points = points.reshape(-1, 3)
    print(f"shape:{points.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    #DBSCAN clustering and visualize
    labels = np.array(pcd.cluster_dbscan(eps=0.018, min_points=20, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0 
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

  
    target_color = np.array([0.12156863, 0.46666667, 0.70588235])  # Light blue in "tab20"

    # Find the label with the color closest to this blue
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1 labels)
    door_label = None
    for label in unique_labels:
        label_color = colors[labels == label][0][:3]  # Get the color for this label
        if np.allclose(label_color, target_color, atol=0.1):  # Allow a small tolerance
            door_label = label
            break

    if door_label is not None:
        # Extract points of the door based on the door_label
        door_points = points[labels == door_label]
        door_pcd = o3d.geometry.PointCloud()
        door_pcd.points = o3d.utility.Vector3dVector(door_points)

        # Calculate the OBB for the refined door points
        obb = door_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Set the OBB color to red
        return pcd,obb
    else:
        print("No label matched the door's color.")
        return pcd, None

   
def get_obb_parameter(obb):

 
    obb_center = obb.center  # Center of the OBB
    obb_extents = obb.extent  # Dimensions (lengths along the principal axes)
    obb_rotation_matrix = obb.R  # Rotation matrix (orientation of the OBB)

    # Print the OBB parameters
    print(f"OBB Center: {obb_center}")
    print(f"OBB Extents (Lengths along axes): {obb_extents}")
    print(f"OBB Rotation Matrix:\n {obb_rotation_matrix}")

def obb_from_rgbd():
   
    depth_image = cv2.imread("data/open_door/episode_1/front_depth/0.png", cv2.IMREAD_UNCHANGED)  # Load depth data
    mask = cv2.imread("data/open_door/episode_1/front_mask/0.png",cv2.IMREAD_UNCHANGED)  # Load mask where door pixels are marked

    # Assume we know the camera intrinsics (fx, fy, cx, cy)
    fx, fy = 175.8385604,175.8385604  # Focal lengths
    cx, cy = 64,64  # Optical center
    
    # Convert masked depth pixels to 3D coordinates (assuming door mask is labeled with 1)
    door_points = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if mask[v, u] == 1:  # Only process door pixels
                z = depth_image[v, u]
                if z > 0:  # Valid depth
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    door_points.append([x, y, z])

    # Convert list to a NumPy array
    door_points = np.array(door_points)

    # Create an Open3D point cloud for the door points
    door_pcd = o3d.geometry.PointCloud()
    door_pcd.points = o3d.utility.Vector3dVector(door_points)

    # Compute the Oriented Bounding Box (OBB) for the door
    obb = door_pcd.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # Set OBB color for visualization

    # Visualize the door point cloud with the OBB
    print("Visualizing door point cloud with OBB...")
    o3d.visualization.draw_geometries([door_pcd, obb])

def filter_with_depth():
    
    # Load point cloud, depth, and mask data
    points = np.load("data/open_door/episode_1/front_pcd/0.npy").reshape(-1, 3)
    depth_image = cv2.imread("data/open_door/episode_1/front_depth/0.png", cv2.IMREAD_UNCHANGED)
    # Camera intrinsic parameters (adjust as needed for your setup)
    
    print(f"depth_image:{depth_image.shape}")
    fx, fy = 175.8385604,175.8385604  # Focal lengths
    cx, cy = 64,64  # Optical center

    # Convert point cloud to an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Filter out points that are occluded based on the depth map
    filtered_points = []
    for point in points:
        x, y, z = point
        u = int((x * fx / z) + cx)  # Project 3D point to 2D pixel u-coordinate
        v = int((y * fy / z) + cy)  # Project 3D point to 2D pixel v-coordinate

        # Check if the point lies within the image boundaries
        # if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
        #     # Only include the point if it matches the depth value in the depth image
        #     print(f"depth_image[v, u]: {depth_image[v, u]}")
            # if abs(z - depth_image[v, u]) < 0.01:  # Threshold for matching depth
            #     filtered_points.append(point)

    # Convert the filtered points to an Open3D point cloud for visualization
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))

    # Visualize the filtered point cloud
    print("Visualizing filtered point cloud (matching camera perspective)...")
    o3d.visualization.draw_geometries([filtered_pcd])

def filter_with_mask():
   
    # Load the point cloud from the .npy file
    points = np.load("data/open_door/episode_1/front_pcd/0.npy").reshape(-1, 3)

    # Load the mask image
    mask_image = cv2.imread("data/open_door/episode_1/front_mask/0.png", cv2.IMREAD_GRAYSCALE)

    # Camera intrinsics (adjust these values to your camera setup)
    fx, fy = 175.8385604,175.8385604  # Focal lengths
    cx, cy = 64,64  # Optical center

    # Filter points using the mask
    filtered_points = []
    for point in points:
        x, y, z = point
        # Project the 3D point to 2D pixel coordinates
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)

        # Check if the 2D projection is within the image boundaries
        if 0 <= u < mask_image.shape[1] and 0 <= v < mask_image.shape[0]:
            # Check if this point corresponds to the door in the mask
            if mask_image[v, u] == 255:  # Assuming door pixels are labeled with 255
                filtered_points.append(point)

    # Convert the filtered points to an Open3D point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))

    # Visualize the filtered point cloud
    print("Visualizing filtered point cloud (door only)...")
    o3d.visualization.draw_geometries([filtered_pcd])

def get_intrinsics():
    # Assuming 'env' is an instance of the class containing get_scene_data
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,JointPosition,EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    env = Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
    
    scene_data = env.get_scene_data()

    # Accessing data for specific cameras
    front_camera_data = scene_data.get("front_camera")
    
    # For example, to access the intrinsics of the front camera
    if front_camera_data is not None:
        front_intrinsics = front_camera_data["intrinsics"]
        front_near_plane = front_camera_data["near_plane"]
        front_far_plane = front_camera_data["far_plane"]
        front_extrinsics = front_camera_data["extrinsics"]

        print("Front Camera Intrinsics:\n", front_intrinsics)
        print("Front Camera Near Plane:", front_near_plane)
        print("Front Camera Far Plane:", front_far_plane)
        print("Front Camera Extrinsics:\n", front_extrinsics)


if __name__ == "__main__":
    iterate()


