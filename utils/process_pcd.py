import numpy as np
import open3d as o3d
from typing import List
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import time
import copy


class PointCloudBase:
    """
    Base class for point cloud operations.
    Provides shared methods for visualization and processing.
    """
    def __init__(self, pcd_dir: str, step: int = 0, label: bool = False):
        """
        Initialize the base class with a list of point cloud files.

        :param pcd_dir: Directory containing point cloud files.
        :param step: Initial step to load point cloud data.
        """
        self.pcd_dir = pcd_dir
        self.label = label
        self.step = step
        self.points = self._load_pcd(step)
        self.pcd = o3d.geometry.PointCloud()
        if not label:
            self.pcd.points = o3d.utility.Vector3dVector(self.points)

    def visualize_pcd(self):
        """
        Visualize a single point cloud.

        :param pcd: Point cloud to visualize.
        """
        if not self.label:
            o3d.visualization.draw_geometries([self.pcd])
        else:
            points = self.points[:, :3] 
            labels = self.points[:, 3]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.zeros((points.shape[0], 3))  # Default color (black)
            colors[labels == 1] = [1, 0, 0]  # Red for closest point
            colors[labels == 0] = [0, 0, 1]  # Blue for other points
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([self.pcd])

    def _load_pcd(self, step: int) -> np.ndarray:
        """
        Load point cloud data for a specific step.

        :param step: Step number to load.
        :return: Numpy array of point cloud data.
        """
        file_name = f"{step}.npy"
        pcd_file = os.path.join(self.pcd_dir, file_name)
        if not os.path.exists(pcd_file):
            raise FileNotFoundError(f"File not found: {pcd_file}")
        return np.load(pcd_file)

    def visualize_pcd_video(self, delay: float = 0.5):
        """
        Visualize point clouds from multiple steps as a video.

        :param steps: Number of steps to visualize.
        :param delay: Time delay (in seconds) between frames.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        filenames = sorted(os.listdir(self.pcd_dir), key=lambda x: int(x.replace('.npy', '')))

        try:
            for filename in filenames:
                step = int(filename.replace('.npy',' '))
                print(f"Step is {step}")
                data = self._load_pcd(step)
                if self.label :
                    points = data[:, :3] 
                    labels = data[:, 3]
                    self.pcd.points = o3d.utility.Vector3dVector(points)
                    colors = np.zeros((points.shape[0], 3))  # Default color (black)
                    colors[labels == 1] = [1, 0, 0]  # Red for closest point
                    colors[labels == 0] = [0, 0, 1]  # Blue for other points

                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    self.pcd.points = o3d.utility.Vector3dVector(data)

                # Update the visualization
                vis.add_geometry(self.pcd)
                vis.update_geometry(self.pcd)
                vis.poll_events()
                vis.update_renderer()

                # Pause for the delay
                time.sleep(delay)
        finally:
            # Ensure the visualizer window is destroyed properly
            vis.destroy_window()

    def _quaternion_to_euler(self, quaternion: list) -> tuple:
        """
        Convert quaternion (qx, qy, qz, qw) to Euler angles (yaw, pitch, roll).
        :param quaternion: List of quaternion values [qx, qy, qz, qw]
        :return: Yaw, Pitch, Roll in degrees
        """
        rotation = R.from_quat(quaternion)
        yaw, pitch, roll = rotation.as_euler('zyx', degrees=True)  # ZYX convention (yaw, pitch, roll)
        return yaw, pitch, roll

class PointCloudFromObservation(PointCloudBase):
    """
    Class for processing point cloud data obtained from observation.
    """
    def __init__(self, pcd_dir: str, step: int = 0):
        super().__init__(pcd_dir,step)

    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        """
        Calculate the oriented bounding box (OBB) of the point cloud.

        :param pcd: Point cloud to calculate the OBB for.
        :return: The calculated oriented bounding box.
        """
        #DBSCAN clustering and visualize
        labels = np.array(self.pcd.cluster_dbscan(eps=0.018, min_points=20, print_progress=True))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0 
        self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
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
            door_points = self.points[labels == door_label]
            door_pcd = o3d.geometry.PointCloud()
            door_pcd.points = o3d.utility.Vector3dVector(door_points)

            # Calculate the OBB for the refined door points
            obb = door_pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0)  # Set the OBB color to red
            return obb
        else:
            print("No label matched the door's color.")
            return None

    def adjust_obb_rotation(self, obb: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
        """
        Adjust the rotation of the OBB to align with the desired orientation.

        :param obb: The oriented bounding box to adjust.
        :param rotation_matrix: The desired rotation matrix.
        """
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

        return obb

    def create_axes_from_obb(self, obb: o3d.geometry.OrientedBoundingBox) -> List[o3d.geometry.LineSet]:
        """
        Create axes for visualization based on the OBB.

        :param obb: The oriented bounding box.
        :return: Axes as numpy arrays (x_axis, y_axis, z_axis).
        """
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
            points=o3d.utility.Vector3dVector([center, center + R[:, 2] * 2*extents[0]]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        z_axis.colors = o3d.utility.Vector3dVector([(0, 0, 1)])  # Blue for z-axis

        return [x_axis, y_axis, z_axis]

    def get_obb_param(self, obb: o3d.geometry.OrientedBoundingBox):

        obb_center = obb.center  # Center of the OBB
        obb_extents = obb.extent  # Dimensions (lengths along the principal axes)
        obb_rotation_matrix = obb.R  # Rotation matrix (orientation of the OBB)

        # Print the OBB parameters
        print(f"OBB Center: {obb_center}")
        print(f"OBB Extents (Lengths along axes): {obb_extents}")
        print(f"OBB Rotation Matrix:\n {obb_rotation_matrix}")

    def visualize_pcd_obb(self):
        """
        Visualize the final calculated and adjusted OBB and pcd.
        """
        obb = self.get_oriented_bounding_box()
        original_obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent)
        obb = self.adjust_obb_rotation(original_obb)
        axes = self.create_axes_from_obb(obb)
        o3d.visualization.draw_geometries([self.pcd,obb,*axes])

    def visualize_pcd_bb_video(self, delay: float = 0.5):
        
        """
        Visualize point clouds from multiple steps as a video.

        :param steps: Number of steps to visualize.
        :param delay: Time delay (in seconds) between frames.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the initial geometry
        vis.add_geometry(self.pcd)
        filenames = sorted(os.listdir(self.pcd_dir), key=lambda x: int(x.replace('.npy', '')))

        try:
            for filename in filenames:
                step = int(filename.replace('.npy',' '))
                print(f"Step is {step}")
                points = self._load_pcd(step)
                points = points.reshape(-1, 3)
                self.pcd.points = o3d.utility.Vector3dVector(points)
                obb = self.get_oriented_bounding_box()
                # Update the visualization
                vis.add_geometry(self.pcd)
                vis.update_geometry(self.pcd)
                vis.add_geometry(obb)
                vis.update_geometry(obb)
                new_obb = copy.deepcopy(obb)
                new_obb = self.adjust_obb_rotation(new_obb)
                vis.poll_events()
                vis.update_renderer()

                # Pause for the delay
                time.sleep(delay)
        finally:
            # Ensure the visualizer window is destroyed properly
            vis.destroy_window()

class PointCloudFromModel(PointCloudBase):
    """
    Class for processing point cloud data obtained from object models.
    """
    def __init__(self, pcd_dir: str, step: int = 0, label:bool = False):
        """
        Initialize the model point cloud class with task data for poses.

        :param pcd_files: List of file paths to point cloud data.
        :param task_data_path: Path to the task data file containing pose information.
        """
        super().__init__(pcd_dir, step, label)
        self.step = step
        self.base_dir = os.path.dirname(pcd_dir)
        self.task_data_path = os.path.join(self.base_dir, 'task_data.npy')
        self.dist_data_path = os.path.join(self.base_dir, 'dist_data.npy')
    

    def extract_poses(self, change: bool = False) -> List[np.ndarray]:
        """
        Extract poses from the task data.

        :return: List of poses as numpy arrays.
        """
        data = np.load(self.task_data_path, allow_pickle=True)
        poses = []
        previous_pose = None
        for i, row in enumerate(data):
            row_list = row.tolist()
            object = 'door_main_visible'  # Adjust as per your data format
            try:
                idx = row_list.index(object)
                current_pose = row_list[idx-7:idx]  # Extract pose preceding the keyword
                if change == False:
                    position = np.array(current_pose[:3], dtype=np.float64)
                    quaternion = np.array(current_pose[3:], dtype=np.float64)
                    poses.append((position, quaternion))
                else:
                    current_orientation = current_pose[-4:]
                    if current_pose != previous_pose:
                        yaw, pitch, roll = self._quaternion_to_euler(current_orientation)
                        print(f"Line {i}: {object} orientation (yaw, pitch, roll): ({yaw:.2f}, {pitch:.2f}, {roll:.2f})")
                        # print(f"Line {i}: {object} pose: {current_pose}")
                        previous_pose = current_pose

            except ValueError:
                print(f"Object 'door_main_visible' not found in row {i}. Skipping...")
                poses.append(None)
        return poses

    def visualize_pcd_pose(self):
        """
        Visualize the point cloud and its corresponding pose as a coordinate frame.

        :param pcd: The point cloud to visualize.
        :param pose: The pose to visualize as a coordinate frame.
        """
        points = np.asarray(self.pcd.points)
        poses = self.extract_poses()
        position, quaternion = poses[self.step]
        rotation = R.from_quat(quaternion).as_matrix()

        # Plot the point cloud and frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, label='Point Cloud')

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
        plt.title(f"Step {self.step}")
        plt.legend()
        plt.show()

    def extract_contact_point(self,save_label = None):
        dist_data = np.load(self.dist_data_path)
       
        # Initialize output
        contact_labels = np.zeros((len(dist_data), 10000), dtype=int)
        results = []
        tolerance = 1e-4
        # Iterate through steps to compare with the consistent reference point
        for step in range(len(dist_data)):  # Assuming one step per pcd_from_mesh file
            # Load pcd_from_mesh for the current step
            points = self._load_pcd(step)
            reference_point = dist_data[step, :3]
            # Check if the reference point exists in the current point cloud
            distances = np.linalg.norm(points - reference_point, axis=1)  
            min_index = np.argmin(distances)  
            labels = np.zeros((points.shape[0], 1), dtype=int)
            labels[min_index] = 1  # Mark the closest point
            contact_labels[step, min_index] = 1
            # Concatenate the points and labels to form an (N, 4) array
            points_with_labels = np.hstack((points, labels))
            closest_point = points[min_index]
            # match_indices = np.where(np.all(np.isclose(points, reference_point, atol=tolerance), axis=1))[0]
            # if min_index > 0:
            results.append((step, min_index))  # Store step index and point index
            # else:
            #     results.append((step, None))  # No match found
            if save_label:
                output_dir = os.path.join(self.base_dir, "pcd_with_labels")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{step}.npy")
                np.save(output_file, points_with_labels)
        # Output results
        for step, result in results:
            print(f"Step {step}: Point Index in PCD = {result}")
        contact_label_file = os.path.join(self.base_dir, "contact_label.npy")
        np.save(contact_label_file, contact_labels)

        #validation of each step
        # for step, result in enumerate(contact_labels):
        #     contact_points_count = np.sum(result)  # Count the number of contact points in this step
        #     print(f"Step {step}: Number of contact points = {contact_points_count}")
        
    def extract_contact_point_two_stage(self, save_label=False):
        """
        Extract contact points with different methods for approaching and contact stages.
        
        Approaching stage: Use the contact point from the first contact step
        Contact stage: Calculate contact points based on minimum distance
        """
        dist_data = np.load(self.dist_data_path)
        data = np.load(self.task_data_path, allow_pickle=True)
        
        # Initialize output
        contact_labels = np.zeros((len(dist_data), 10000), dtype=int)
        results = []
        
        # First, detect the transition step where door starts opening
        transition_step = None
        previous_orientation = None
        
        for i, row in enumerate(data):
            row_list = row.tolist()
            object_name = 'door_main_visible'
            
            try:
                idx = row_list.index(object_name)
                current_pose = row_list[idx-7:idx]
                current_orientation = current_pose[-4:]  # Extract quaternion
                
                if previous_orientation is not None:
                    # Check if orientation has changed (door has started opening)
                    if not np.array_equal(current_orientation, previous_orientation):
                        transition_step = i
                        print(f"Door opening detected at step {transition_step}")
                        break
                
                previous_orientation = current_orientation
            except ValueError:
                print(f"Object '{object_name}' not found in row {i}. Skipping...")
        
        if transition_step is None:
            print("Warning: Could not detect door opening transition. Using default approach.")
            return self.extract_contact_point(save_label)
        
        # Get contact point at transition step
        transition_points = self._load_pcd(transition_step)
        reference_point = dist_data[transition_step, :3]
        distances = np.linalg.norm(transition_points - reference_point, axis=1)
        transition_contact_idx = np.argmin(distances)
        transition_contact_point = transition_points[transition_contact_idx]
        
        print(f"Transition contact point found at index {transition_contact_idx}")
        
        # Process each step based on its stage
        for step in range(len(dist_data)):
            points = self._load_pcd(step)
            
            # Approaching stage: use same contact point as transition step
            if step < transition_step:
                # Find the point in current PCD closest to the transition contact point
                distances = np.linalg.norm(points - transition_contact_point, axis=1)
                contact_idx = np.argmin(distances)
                print(f"Approaching stage - Step {step}: using transition point as reference")
            
            # Contact stage: calculate based on minimum distance to reference point
            else:
                reference_point = dist_data[step, :3]
                distances = np.linalg.norm(points - reference_point, axis=1)
                contact_idx = np.argmin(distances)
                print(f"Contact stage - Step {step}: using min distance calculation")
            
            # Create labels
            labels = np.zeros((points.shape[0], 1), dtype=int)
            labels[contact_idx] = 1
            contact_labels[step, contact_idx] = 1
            results.append((step, contact_idx))
            
            # Save labeled point cloud if requested
            if save_label:
                points_with_labels = np.hstack((points, labels))
                output_dir = os.path.join(self.base_dir, "pcd_with_labels")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{step}.npy")
                np.save(output_file, points_with_labels)
        
        # Output results
        for step, result in results:
            print(f"Step {step}: Point Index in PCD = {result}")
        
        # Save contact labels
        contact_label_file = os.path.join(self.base_dir, "contact_label.npy")
        np.save(contact_label_file, contact_labels)
        
        return results, contact_labels

    def print_dist(self):
        dist_data = np.load(self.dist_data_path)
        for step in range(len(dist_data)): 
            print(f"Step {step}: {dist_data[step, -1]* 1000}")




if __name__ == "__main__":

    # pcd1 = PointCloudFromModel(
    #     pcd_dir = "data/open_door/episode_0/pcd_with_labels", 
    #     step=1, label = True)
    
    # pcd1.visualize_pcd()

    # pcd1 = PointCloudFromModel(
    #     pcd_dir = "data/open_door/episode_0/pcd_from_mesh", 
    #     step=1)
    
    # pcd1.extract_poses(change=True)
    # pcd1.extract_contact_point()



    ###### produce contact label
    base_dir = "data/open_door"
    episode_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('episode')]
    for episode_dir in episode_dirs:
        print(f"Processing {episode_dir}...")
        pcd_dir = os.path.join(base_dir, episode_dir, "pcd_from_mesh")
        pcd = PointCloudFromModel(
            pcd_dir=pcd_dir,
            step=1,  # Assuming step = 1 for the example, you can adjust as needed
            label=True
        )
        pcd.extract_contact_point_two_stage()

    print("All episodes processed.")