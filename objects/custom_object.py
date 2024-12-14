from pyrep.objects.object import Object
import numpy as np
from typing import Tuple
from pyrep.backend import sim

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

        # # Extract the distance and closest points from the return value
        # obj1_point = np.array(distance_data[:3], dtype=np.float64)  # [obj1X, obj1Y, obj1Z]
        # obj2_point = np.array(distance_data[3:6], dtype=np.float64)  # [obj2X, obj2Y, obj2Z]
        # distance = distance_data[6]  # Minimum distance between objects

        return distance_data
        # return sim.simCheckDistance(
        #     self.get_handle(), other.get_handle(), -1)[6]
