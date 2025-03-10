import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep import PyRep
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from scipy.spatial.transform import Rotation as R

class VisualizeEpisode:
    def __init__(self, saved_data_path):
        self.saved_data = np.load(saved_data_path, allow_pickle=True)
        self.predicted_targets = []
        self._initialize_environment()

    def _initialize_environment(self):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=False
        )
        self.env.launch()
        from rlbench.backend.utils import task_file_to_task_class
        task_name = task_file_to_task_class("open_door")
        self.task_env = self.env.get_task(task_name)

    def rotation_6d_to_quaternion(self, rot_6d):
        v1 = rot_6d[:3]  # First column
        v2 = rot_6d[3:]  # Second column

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)

        rot_matrix = np.stack([v1, v2, v3], axis=-1)  # 3x3 matrix
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()
        return quat  # Quaternion in [x, y, z, w] format


    def visualize_step(self, t, actions, r_door, contact_position):
        # Create or update the reference frame
        from pyrep import PyRep



        if t == 0:
            ref_frame = Dummy.create(size=0.05)
            pr = PyRep()
            # ref_frame = pr.import_model("/home/tongmiao/CoppeliaSim/models/other/reference frame.ttm")
            ref_frame.set_name(f"Reference_Frame_{t}")
        else:
            ref_frame = Dummy(f"Reference_Frame_{t-1}")
            ref_frame.set_name(f"Reference_Frame_{t}")
        
        ref_frame.set_position(contact_position)
        ref_frame.set_quaternion(r_door)

        for i, action in enumerate(actions):
            quat = self.rotation_6d_to_quaternion(action[3:9])

            if t == 0 :
                # predicted_target = Dummy.create(size=0.005)
                predicted_target = pr.import_model("/home/tongmiao/CoppeliaSim/models/other/reference frame.ttm")

                # predicted_target = Shape.create(type=sim.sim_primitiveshape_sphere, 
                #                             size=[0.015, 0.015, 0.015],  # Smaller sphere
                #                             color=[1, 0, 0])
                predicted_target.set_name(f"Predicted_EEF_Target_{i}")
            else:
                predicted_target = Dummy(f"Predicted_EEF_Target_{i}")
            
            # Set position and quaternion relative to the reference frame
            predicted_target.set_position(action[:3], relative_to=ref_frame)
            predicted_target.set_quaternion(quat, relative_to=ref_frame)
            
        self.env._pyrep.step()
        print(f"Visualized step {t} with {len(actions)} sequential actions relative to ref_frame.")

if __name__ == "__main__":
    saved_data_path = "data/open_door/episode_0/validate_training_input_obc.npy"  # Update with correct path
    visualizer = VisualizeEpisode(saved_data_path)
    for t, step_data in enumerate(visualizer.saved_data):
        visualizer.visualize_step(t, step_data['action'], step_data['pose'], step_data['point'])
