import time
import os
import torch
import numpy as np
from PIL import Image
from rlbench.backend.const import *
# import matplotlib.pyplot as plt
import argparse
from rlbench.backend import utils
# from RobotIL.constants import DT
# from utils.utils import set_seed
# from RobotIL.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
# from scipy.spatial.transform import Rotation as R
# import numpy as np
from inferenceAPI import PolicyInferenceAPI
from scipy.spatial.transform import Rotation as R
class SEILinference(PolicyInferenceAPI):

    def __init__(self,config):
        super().__init__(config)


    def _initialize_environment(self):
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,JointPosition,EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
       
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        if self.config["predict_value"] == "ee_pos_ori":
            self.env = Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
            print("EndEffectorPoseViaPlanning")
        else:
            self.env = Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
        
        self.env.launch()
        from rlbench.backend.utils import task_file_to_task_class
        task_name = task_file_to_task_class(self.config["task_name"])
        self.task_env = self.env.get_task(task_name)
        

    def _get_data(self, t):
        # rgb_images = []
        if t == 0:
            descriptions, obs = self.task_env.reset()
            print(f"reset successfully")
        else:
            obs = self.task_env.get_observation()

        task_data = np.array(obs.task_low_dim_state)
        object_name = 'door_main_visible'
        door_pose = None

        # Convert the entire row to a Python list
        task_data_list = task_data.tolist()

        try:
            # Find the index of the door object name
            idx = task_data_list.index(object_name)

            # Extract the 7 elements right before the object name
            # [position_x, position_y, position_z, quat_x, quat_y, quat_z, quat_w]
            current_pose = task_data_list[idx - 7 : idx]

            # The first 3 are the position; the last 4 are the quaternion
            position = np.array(current_pose[:3], dtype=np.float64)
            quaternion = np.array(current_pose[3:], dtype=np.float64)
            door_pose = np.concatenate([position,quaternion])
            door_pose = np.array(door_pose)
            door_pose = torch.from_numpy(door_pose).float().cuda().unsqueeze(0)
            # print(f"Door pose {door_pose} at timestep {t}'s shape:{door_pose.shape}")

        except ValueError:
            # If 'door_main_visible' is not found, we just skip it
            print(f"Object '{object_name}' not found in the current observation. Skipping...")

        pcd_from_mesh = np.array(obs.pcd_from_mesh)
        # print(f"Shape of pcd_from_mesh at timestep {t} during inference : {pcd_from_mesh.shape}")

        if self.config["predict_value"] == "joint_states":
            gripper_open = np.array(obs.gripper_open).reshape(1)
            joint_positions = np.array(obs.joint_positions)
            qpos = np.concatenate([joint_positions,gripper_open])
            qpos = np.array(qpos)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            print(f"Shape of qpos at timestep {t}during inference : {qpos.shape}")
        else:
            gripper_open = np.array(obs.gripper_open).reshape(1)
            gripper_pose = np.array(obs.gripper_pose)
            ee_pos = gripper_pose[:3]
            ee_quat = gripper_pose[3:7]
            ee_rot_6d = self.quaternion_to_6d(ee_quat)
            gripper_states = np.concatenate([ee_pos,ee_rot_6d, gripper_open])
            qpos_numpy = np.array(gripper_states)
            qpos = self._pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            # print(f"Shape of qpos at timestep {t}during inference : {qpos.shape}") [1,10]

        if self.config["obs_type"] == "rgbd":
            # right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
            right_shoulder_rgb = obs.right_shoulder_rgb
            right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
            right_shoulder_rgb = np.array(right_shoulder_rgb)
            right_shoulder_depth = np.array(right_shoulder_depth)
            # print(f"Initial shape of right_shoulder_rgb: {right_shoulder_rgb.shape}")
            # print(f"Initial shape of right_shoulder_depth: {right_shoulder_depth.shape}")
            #128*128*3
            rgb_list = [right_shoulder_rgb,right_shoulder_depth]
            curr_image = np.array(rgb_list)
            curr_image = torch.from_numpy(curr_image).float() / 255.0
            # curr_image = (
            #     curr_image.permute(0, 3, 1, 2).view((1, -1, 3, 128, 128)).cuda()
            # )
            curr_image = curr_image.permute(0, 3, 1, 2).unsqueeze(0).cuda()
            rgb_images = curr_image
            # rgb_images.append(curr_image)
        
        elif self.config["obs_type"] == "pcd":
            front_point_cloud = np.array(obs.front_point_cloud)
            pcd = torch.from_numpy(front_point_cloud).float()
            pcd = pcd.permute(2, 0, 1)
            pcd = pcd.unsqueeze(0)
            rgb_images = pcd
            # rgb_images.append(pcd)

        return qpos, rgb_images, door_pose, pcd_from_mesh
        
    def quaternion_to_matrix(self, quat):
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
    
    def _run(self, qpos, rgb_images, t, all_time_actions=None,door_pose=None,pcd_from_mesh=None):
        """
        Predicts and executes actions based on collected data (qpos and images).

        Args:
            qpos_history (torch.Tensor): A tensor containing joint positions.
            rgb_images (list): A list of RGB images.
            max_timesteps (int): Maximum number of timesteps to run the simulation.
            all_time_actions (torch.Tensor, optional): Stores actions for temporal aggregation.
            door_pose (torch.Tensor): Shape (1, 7) => [x, y, z, q_x, q_y, q_z, q_w]
            pcd_from_mesh (torch.Tensor or np.ndarray): Shape (num_points, 3), in world coords
        Returns:
            None
        """
        curr_image = rgb_images
        # pre_contact : shape [1,1,10000]
        action, pred_contact, actions = self._query_policy(t, qpos, curr_image, all_time_actions)

        if self.config["predict_value"] == "ee_pos_ori":

            pred_contact = pred_contact.squeeze().detach().cpu().numpy()  # shape [10000]
            contact_idx = np.argmax(pred_contact)
            contact_point_position = pcd_from_mesh[contact_idx]

            door_pose_np = door_pose.squeeze(0).detach().cpu().numpy()  # shape (7,)
            door_quat = door_pose_np[3:]            # (q_x, q_y, q_z, q_w)
            if t % 5 ==0:
                self.visualize_step(t, actions, door_quat, contact_point_position)
            R_door = self.quaternion_to_matrix(door_quat)
            
            action = action.squeeze(0).cpu().numpy()  # No need to detach here
            action_position = action[:3]
            action_6d = action[3:9]
            predicted_gripper = np.array(action[-1])

            action_world_pos = (contact_point_position + R_door.dot(action_position))

            action_world_rot = R_door.dot(action_6d.reshape(3, 2))
            action_world_rot = action_world_rot.T.reshape(-1)
            action_world_quat = self.rotation_6d_to_quaternion(action_world_rot)
            norm = np.linalg.norm(action_world_quat)
            if norm == 0:
                raise ValueError("The quaternion has zero magnitude and cannot be normalized.")
            action_world_quat = action_world_quat / norm
            action_world = np.concatenate([action_world_pos, action_world_quat,[predicted_gripper]], axis=-1)
            # action_world = torch.from_numpy(action_world).float().cuda().unsqueeze(0)
            # action_world = action_world.detach().cpu().numpy()
            # self.task_env.step(action_world)
            # self.test_by_dummy(action_world, t)

        else:
            self.task_env.step(action)

    def test_by_collect(self,t):
        action = np.load(
            f"data/{self.config['task_name']}/episode_0/gripper_states.npy"
        ).astype(np.float32)
        action = action[t,:]
        print(f"{t} step action is {action}")
        self.task_env.step(action)
    
    def test_by_dummy(self,action,t):
        from pyrep.objects.dummy import Dummy
        # from pyrep import PyRep
        if t == 0 : 
            predicted_target = Dummy.create(size=0.05)
            predicted_target.set_name('Predicted_EEF_Target')
        else :
            predicted_target = Dummy('Predicted_EEF_Target')
        predicted_target.set_position(action[:3])
        predicted_target.set_quaternion(action[3:-1])
        self.task_env._pyrep.step()
        print(f"{t} dummy action is {action}")

    def visualize_step(self, t, actions, r_door, contact_position):
        from pyrep.objects.dummy import Dummy
        from pyrep import PyRep
        # Create or update the reference frame
        if t == 0:
            pr = PyRep()
            ref_frame = Dummy.create(size=0.05)
            ref_frame.set_name(f"Reference_Frame_{t}")
        else:
            ref_frame = Dummy(f"Reference_Frame_{t-5}")
            ref_frame.set_name(f"Reference_Frame_{t}")
        
        ref_frame.set_position(contact_position)
        ref_frame.set_quaternion(r_door)
        # ref_frame.set_visibility(False)
        actions = actions.squeeze(0).detach().cpu().numpy()
        print(f"actions shape is {actions.shape}")

        for i, action in enumerate(actions[:5]):
            print(f"Action shape is {action.shape}")
            quat = self.rotation_6d_to_quaternion(action[3:9])

            if t == 0 :
                # predicted_target = Dummy.create(size=0.05)
                predicted_target = pr.import_model("../CoppeliaSim/models/other/reference frame.ttm")

                predicted_target.set_name(f"Predicted_EEF_Target_{i}")
            else:
                predicted_target = Dummy(f"Predicted_EEF_Target_{i}")
            
            # Set position and quaternion relative to the reference frame
            predicted_target.set_position(action[:3], relative_to=ref_frame)
            predicted_target.set_quaternion(quat, relative_to=ref_frame)

        self.task_env._pyrep.step()
        print(f"Visualized step {t} with {len(actions)} sequential actions relative to ref_frame.")


    def quaternion_to_6d(self, q):
        # Convert quaternion to rotation matrix
        r = R.from_quat(q)
        rot_matrix = r.as_matrix()  # 3x3 matrix

        # Take the first two columns
        m1 = rot_matrix[:, 0]
        m2 = rot_matrix[:, 1]

        # Flatten to 6D vector
        rot_6d = np.concatenate([m1, m2], axis=-1)
        return rot_6d
    

    def _render_step(self):
        pass

def parse_arguments():

    parser = argparse.ArgumentParser(description="Policy Inference")

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint",
    )
    parser.add_argument(
        "--ckpt_name", type=str, default="policy_best.ckpt", help="Checkpoint file name"
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task to evaluate"
    )
    parser.add_argument(
        "--policy_class",
        type=str,
        default="ACT",
        choices=["ACT", "CNNMLP", "Diffusion"],
        help="Class of the policy to use",
    )
    parser.add_argument(
        "--visual_encoder", type=str, default="dinov2", help="Type of visual encoder"
    )
    parser.add_argument(
        "--variant", type=str, default="vits14", help="Variant of the visual encoder"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--temporal_agg",
        action="store_true",
        help="Enable temporal aggregation for ACT policy",
    )
    parser.add_argument(
        "--predict_value", type=str, default="ee_pos_ori", help="Value to predict"
    )
    parser.add_argument(
        "--obs_type", type=str, default="rgbd", help="rgbd or depth"
    )
    parser.add_argument(
        "--episode_len", type=int, default=300, help="Maximum length of each episode"
    )
    # For ACT
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL Weight")
    parser.add_argument(
        "--contact_weight", type=float, default=1.0, help="Contact loss weight"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100, help="Number of queries (chunk size)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension size"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=3200, help="Feedforward dimension size"
    )
    parser.add_argument(
        "--enc_layers", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--dec_layers", type=int, default=7, help="Number of decoder layers"
    )
    parser.add_argument(
        "--nheads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for evaluation"
    )
    
    parser.add_argument(
        "--state_dim", type=int, default=10, help="state_dimension"
    )
    return vars(parser.parse_args())

def main():
    
    args = parse_arguments()

    
    # Prepare configuration dictionary
    config = {
    "env_class":None,#TODO
    "ckpt_dir": os.path.join(args["ckpt_dir"], args["task_name"]),
    "ckpt_name": args["ckpt_name"],
    
    "policy_class": args["policy_class"],
    "policy_config": {
        "lr": args["lr"],  # Learning rate for evaluation
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "contact_weight": args["contact_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": 1e-5,  # As per the correct code
        "backbone": args["visual_encoder"],
        "variant": args["variant"],
        "enc_layers": args["enc_layers"],
        "dec_layers": args["dec_layers"],
        "nheads": args["nheads"],
        "camera_names": ["top"],  # Assuming camera_names is ["top"]
        "state_dim":  args["state_dim"],  # TODO
    },
    "task_name": args["task_name"],
    "seed": args["seed"],
    "temporal_agg": args["temporal_agg"] if args["policy_class"] == "ACT" else False,
    "predict_value": args["predict_value"],
    "obs_type": args["obs_type"],
    "batch_size": 1,
    "episode_len": args["episode_len"],
    "num_epochs": 1,  # Default number of epochs for evaluation
    }

    # Initialize the PolicyInferenceAPI
    inference = SEILinference(config)
    try:
        # Execute the inference
        inference.run_inference(ckpt_name=args["ckpt_name"])
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    main()