from typing import List, Tuple
import numpy as np
from objects.custom_object import CustomObject
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from scipy.spatial.transform import Rotation as R
from objects.custom_object import MeshBase

class OpenDoor(Task):

    def init_task(self) -> None:
        self._door_joint = Joint('door_frame_joint')
        self.register_success_conditions([
            JointCondition(self._door_joint, np.deg2rad(25))])
        self._door_unlock_cond = JointCondition(
            Joint('door_handle_joint'), np.deg2rad(25))

    def init_episode(self, index: int) -> List[str]:
        self._door_unlocked = False
        self._door_joint.set_motor_locked_at_zero_velocity(True)
        return ['open the door',
                'grip the handle and push the door open',
                'use the handle to open the door']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        if not self._door_unlocked:
            self._door_unlocked = self._door_unlock_cond.condition_met()[0]
            if self._door_unlocked:
                self._door_joint.set_motor_locked_at_zero_velocity(False)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
    
    def get_door_pcd(self) -> np.ndarray:
        self._door = CustomObject(Object.get_object('door_main_visible'))
        if not self._door:
            raise ValueError("Door object 'door_main_visible' not found in the simulation.")
        vertices, indices, normals = self._door.get_mesh()
        self.mesh_base = MeshBase(vertices, indices, normals)
        pose = np.array(self._door._object.get_pose())  # [x, y, z, qx, qy, qz, qw]
        # position = pose[:3]
        # quaternion = pose[3:]

        # # Compute transformation matrix
        # rotation_matrix = R.from_quat(quaternion).as_matrix()
        # transform_matrix = np.eye(4)
        # transform_matrix[:3, :3] = rotation_matrix
        # transform_matrix[:3, 3] = position

        # # Transform vertices from local to world frame
        # door_pcd = []
        # for i in range(0, len(vertices), 3):
        #     local_point = np.array([vertices[i], vertices[i+1], vertices[i+2], 1.0])  # Homogeneous coordinates
        #     world_point = transform_matrix @ local_point
        #     door_pcd.append(world_point[:3])

        # return np.array(door_pcd, dtype=np.float64)
        num_samples = 10000
        return self.mesh_base.generate_pcd(pose, num_samples=num_samples)
    
    def check_door_distance(self, other:'Object') -> list:
        self._door = CustomObject(Object.get_object('door_main_visible'))
        if not self._door:
            raise ValueError("Door object 'door_main_visible' not found in the simulation.")
        distance_data = self._door.check_distance(other)
        return distance_data
       