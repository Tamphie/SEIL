import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from RobotIL.constants import DT
from utils.utils import set_seed
from RobotIL.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from scipy.spatial.transform import Rotation as R
import numpy as np


class PolicyInferenceAPI:
    """
    A class to perform inference on trained policy models in a specified environment.
    """

    def __init__(self, config):
        """
        Initializes the PolicyInferenceAPI with the given configuration.

        Args:
            config (dict): Configuration dictionary containing all necessary parameters.
        """
        self.config = config
        self.all_actions = None  # Initialize all_actions
        self.set_up()

    def set_up(self):
        """Sets up the inference process by loading the policy and initializing the environment."""
        self._initialize_seed()
        self._load_policy()
        self._initialize_environment()

    def _initialize_seed(self):
        """Sets the random seed for reproducibility."""
        set_seed(self.config["seed"])

    def _make_policy(self):
        """Creates a policy instance based on the policy class and configuration."""
        policy_class = self.config["policy_class"]
        policy_config = self.config["policy_config"]
        if policy_class == "ACT":
            policy = ACTPolicy(policy_config)
        elif policy_class == "CNNMLP":
            policy = CNNMLPPolicy(policy_config)
        elif policy_class == "Diffusion":
            policy = DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError(
                f"Policy class '{policy_class}' is not implemented."
            )
        return policy

    def _load_policy(self, ckpt_name=None):
        """
        Loads the policy from the specified checkpoint.

        Args:
            ckpt_name (str, optional): Name of the checkpoint to load. Defaults to None,
                                        which uses the checkpoint specified in config.
        """
        ckpt_dir = self.config["ckpt_dir"]
        ckpt_name = ckpt_name or self.config.get("ckpt_name", "policy_best.ckpt")
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        self.policy = self._make_policy()
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda:0"))
            loading_status = self.policy.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint '{ckpt_path}' with status: {loading_status}")
        else:
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist.")

        self.policy.cuda()
        self.policy.eval()
        print(f"Policy loaded from: {ckpt_path}")

    def _initialize_environment_pre(self):
        """Initializes the environment based on the task configuration."""
        from RobotIL.core.real_robot.fetch import (
            FetchEnv,
        )  # Imported here to avoid unnecessary dependencies

        task_name = self.config["task_name"]
        predict_value = self.config["predict_value"]
        self.env = FetchEnv(task_name, predict_value=predict_value)

        self.query_frequency = 5
        self.temporal_agg = self.config.get("temporal_agg", False)
        if self.temporal_agg and self.config["policy_class"] == "ACT":
            self.num_queries = self.config["policy_config"]["num_queries"]

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
        
        self.temporal_agg = self.config.get("temporal_agg", False)
        if self.temporal_agg and self.config["policy_class"] == "ACT":
            self.num_queries = self.config["policy_config"]["num_queries"]
        

    def _pre_process(self, s_qpos):
        """Pre-processes the joint positions."""
        return s_qpos

    def _render_step(self):
        """Updates the rendering window with the latest frame."""
        image = self.env._physics.render(height=480, width=640, camera_id="angle")
        self.plt_img.set_data(image)
        plt.pause(DT)

    def _query_policy(self, t, qpos, curr_image, all_time_actions=None):
        """Queries the policy to get the next action."""
        if self.config["policy_class"] == "ACT":
            if t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                if self.all_actions is None:
                    raise ValueError(
                        "all_actions is None when temporal_agg is enabled."
                    )

                all_time_actions[[t], t : t + self.num_queries] = self.all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.1
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
            else:
                if self.all_actions is None:
                    raise ValueError(
                        "all_actions is None when temporal_agg is disabled."
                    )
                raw_action = self.all_actions[:, t % self.query_frequency]
        else:
            # For CNNMLP or Diffusion, temporal aggregation is ignored
            raw_action = self.policy(qpos, curr_image)

        return raw_action

    def get_data(self, max_timesteps):
        """
        Collects data from the environment, including observations, joint positions, and images.

        Args:
            max_timesteps (int): Maximum number of timesteps to run the simulation.

        Returns:
            tuple: A tensor of joint positions (qpos_history) and a list of RGB images.
        """
        qpos_history = torch.zeros((1, max_timesteps, self.config["state_dim"])).cuda()
        rgb_images = []

        for t in range(max_timesteps):
            rgb_list, joints = self.env.get_obs_joint()
            #TODO
            qpos_numpy = np.array(joints)
            qpos = self._pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos

            curr_image = np.array(rgb_list)
            curr_image = torch.from_numpy(curr_image) / 255.0
            curr_image = (
                curr_image.permute(0, 3, 1, 2).view((1, -1, 3, 480, 640)).cuda()
            )
            rgb_images.append(curr_image)

        return qpos_history, rgb_images

    def run_pre(self, qpos_history, rgb_images, max_timesteps, all_time_actions=None):
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
        with torch.no_grad():  # Ensure no gradients are tracked
            for t in range(max_timesteps):
                qpos = qpos_history[:, t]
                curr_image = rgb_images[t]

                action = self._query_policy(t, qpos, curr_image, all_time_actions)
                action = action.squeeze(0).cpu().numpy()  # No need to detach here

                if self.config["predict_value"] == "ee_pos_ori":
                    self.env.ee_space_control(action)
                else:
                    self.env.step(action)



    def rotation_6d_to_quaternion(self, rot_6d):
        """
        Convert a 6D rotation representation back to a quaternion.
        
        :param rot_6d: 6D rotation vector (concatenation of first two columns of rotation matrix)
        :return: Quaternion [w, x, y, z]
        """
        # Split the 6D vector into two 3D vectors (first two columns of the rotation matrix)
        m1 = rot_6d[:3]  # First column
        m2 = rot_6d[3:]  # Second column

        # Compute the third column by taking the cross product of m1 and m2
        # This ensures that the third column is orthogonal to the first two
        m3 = np.cross(m1, m2)

        # Reconstruct the full 3x3 rotation matrix
        rot_matrix = np.stack([m1, m2, m3], axis=-1)  # 3x3 matrix

        # Convert the rotation matrix back to a quaternion
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # Quaternion in [x, y, z, w] format

        # Return quaternion in [w, x, y, z] format for consistency
        return np.array([quat[3], quat[0], quat[1], quat[2]])  
        # TODO verify the quaternion is [w, x, y, z]

    def action_process(self, action):
        predicted_pos = action[:3]
        predicted_rot_6d = action[3:-1]
        predicted_gripper = action[-1]
        predicted_quat = self.rotation_6d_to_quaternion(predicted_rot_6d)

        norm = np.linalg.norm(predicted_quat)
        if norm == 0:
            raise ValueError("The quaternion has zero magnitude and cannot be normalized.")
        quat = predicted_quat / norm
        action_all = np.concatenate([predicted_pos, quat,predicted_gripper], axis=-1)
        return action_all
    

    def run(self, qpos_history, rgb_images, max_timesteps, all_time_actions=None):
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
        from rlbench.backend.utils import task_file_to_task_class
        task_name = task_file_to_task_class(self.config["task_name"])
        task_env = self.env.get_task(task_name)
        descriptions, _ = task_env.reset()
        with torch.no_grad():  # Ensure no gradients are tracked
            
            for t in range(max_timesteps):
                qpos = qpos_history[:, t]
                curr_image = rgb_images[t]

                action = self._query_policy(t, qpos, curr_image, all_time_actions)
                action = action.squeeze(0).cpu().numpy()  # No need to detach here
                
                if self.config["predict_value"] == "ee_pos_ori":
                    action = self.action_process(action)
                    self.env.step(action)
                else:
                    self.env.step(action)                

    def run_inference(self, ckpt_name=None, save_episode=False):
        """
        Runs the policy for a single episode by collecting data and running actions.

        Args:
            ckpt_name (str, optional): Name of the checkpoint to evaluate. If None, uses the loaded policy's checkpoint.
            save_episode (bool, optional): Whether to save episode data.

        Returns:
            None
        """
        if ckpt_name:
            self._load_policy(ckpt_name)

        max_timesteps = self.config["episode_len"]
        temporal_agg = (
            self.temporal_agg if self.config["policy_class"] == "ACT" else False
        )
        num_queries = self.num_queries if temporal_agg else None

        start_time = time.time()

        self.env.seed(0)

        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, self.config["state_dim"]]
            ).cuda()

        

        # Collect data from the environment
        qpos_history, rgb_images = self.get_data(max_timesteps)
        #TODO
        # Run the collected data through the policy
        self.run(
            qpos_history,
            rgb_images,
            max_timesteps,
            all_time_actions if temporal_agg else None,
        )

        print(f"Inference took {time.time() - start_time:.2f} seconds")

        # Save the episode if necessary
        if save_episode:
            result_file_name = (
                f"result_{ckpt_name.split('.')[0]}.txt" if ckpt_name else "result.txt"
            )
            with open(
                os.path.join(self.config["ckpt_dir"], result_file_name), "w"
            ) as f:
                f.write("Inference completed.\n")


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
    "ckpt_dir": os.path.join(args["ckpt_dir"], args["task_name"]),
    "ckpt_name": args["ckpt_name"],
    "state_dim": 20,  # Assuming state_dim of 20, adjust as needed
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
    },
    "task_name": args["task_name"],
    "seed": args["seed"],
    "temporal_agg": args["temporal_agg"] if args["policy_class"] == "ACT" else False,
    "predict_value": args["predict_value"],
    "batch_size": 1,
    "episode_len": args["episode_len"],
    "num_epochs": 1,  # Default number of epochs for evaluation
    }

    # Initialize the PolicyInferenceAPI
    inference_api = PolicyInferenceAPI(config)

    # Execute the inference
    inference_api.run_inference(ckpt_name=args["ckpt_name"])
    print(f"Inference complete for checkpoint: {args["ckpt_name"]}")

if __name__ == "__main__":
    main()