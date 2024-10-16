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
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
       
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        if self.config["predict_value"] == "ee_pos_ori":
            self.env = Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
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
        else:
            obs = self.task_env.get_observation()
        
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

        return qpos, rgb_images
        

    def _run(self, qpos, rgb_images, t, all_time_actions=None):
        """
        Predicts and executes actions based on collected data (qpos and images).

        Args:
            qpos_history (torch.Tensor): A tensor containing joint positions.
            rgb_images (list): A list of RGB images.
            max_timesteps (int): Maximum number of timesteps to run the simulation.
            all_time_actions (torch.Tensor, optional): Stores actions for temporal aggregation.

        Returns:
            None
        """
        curr_image = rgb_images

        action = self._query_policy(t, qpos, curr_image, all_time_actions)
        action = action.squeeze(0).cpu().numpy()  # No need to detach here
        
        if self.config["predict_value"] == "ee_pos_ori":
            action = self.action_process(action)
            #TODO determin if action satisfy:  
            # def act(self, obs):
            # arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
            # gripper = [1.0]  # Always open
            # return np.concatenate([arm, gripper], axis=-1)

            obs, reward, terminate = self.task_env.step(action)
        else:
            obs, reward, terminate = self.task_env.step(action)


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
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": 1e-5,  # As per the correct code
        "backbone": args["visual_encoder"],
        "variant": args["variant"],
        "enc_layers": args["enc_layers"],
        "dec_layers": args["dec_layers"],
        "nheads": args["nheads"],
        "camera_names": ["top"],  # Assuming camera_names is ["top"]
        "state_dim": 10,  # TODO
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

    # Execute the inference
    inference.run_inference(ckpt_name=args["ckpt_name"])
    print(f"Inference complete for checkpoint: {args['ckpt_name']}")


if __name__ == "__main__":
    main()