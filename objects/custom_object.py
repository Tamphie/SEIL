from pyrep.objects.object import Object
import numpy as np
from typing import Tuple
from pyrep.backend import sim
import open3d as o3d
from scipy.spatial.transform import Rotation as R
class CustomObject:
    def __init__(self,object_handle: Object):
          self._object = object_handle

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Get the mesh data (vertices, indices, normals) of this object.

            :return: A tuple containing:
                    - vertices (np.ndarray): The vertices of the mesh.
                    - indices (np.ndarray): The indices of the mesh faces.
                    - normals (np.ndarray): The normals of the mesh.
            """
            vertices, indices, normals = sim.simGetShapeMesh(self._object.get_handle())
            vertices = np.array(vertices, dtype=np.float64)
            indices = np.array(indices, dtype=np.int32)
            normals = np.array(normals, dtype=np.float64)
            return vertices, indices, normals
    def check_distance(self, other: 'Object') -> list:
        """
        Checks the minimum distance between two objects and retrieves the closest points.

        :param other: The other object to check distance against.
        :return: A tuple containing:
                - The minimum distance (float)
                - The closest point on the current object (np.ndarray of [x, y, z])
                - The closest point on the other object (np.ndarray of [x, y, z])
        """
        distance_data = sim.simCheckDistance(self._object.get_handle(), other.get_handle(), -1)
        distanceSegment = sim.simAddDrawingObject(sim.sim_drawing_lines, 4, 0, -1, 1, [0, 1, 0])

        if distance_data[-1] > 0:
                sim.simAddDrawingObjectItem(distanceSegment, None)
                sim.simAddDrawingObjectItem(distanceSegment, distance_data[:6])
        # # Extract the distance and closest points from the return value
        # obj1_point = np.array(distance_data[:3], dtype=np.float64)  # [obj1X, obj1Y, obj1Z]
        # obj2_point = np.array(distance_data[3:6], dtype=np.float64)  # [obj2X, obj2Y, obj2Z]
        # distance = distance_data[6]  # Minimum distance between objects

        return distance_data
        # return sim.simCheckDistance(
        #     self.get_handle(), other.get_handle(), -1)[6]
class MeshBase:
    def __init__(self, vertices, indices, normals):
        """
        Initialize the MeshBase class with vertices, indices, and normals.
        
        Parameters:
           vertices (np.ndarray): The vertices of the mesh.
            indices (np.ndarray): The indices of the mesh faces.
            normals (np.ndarray): The normals of the mesh.
        """
        self.vertices = vertices
        self.indices = indices
        self.normals = normals
    def generate_pcd(self, pose, num_samples = 10000):
        vertices = np.array(self.vertices).reshape(-1, 3)
        indices = np.array(self.indices).reshape(-1, 3)
        
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(indices)
        )

        # Sample points from the mesh
        pcd = mesh.sample_points_uniformly(number_of_points=num_samples)

        # Transform to world frame
        position = pose[:3]
        quaternion = pose[3:]
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Construct transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position

        # Apply transformation
        pcd.transform(transform_matrix)

        return np.asarray(pcd.points)