import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from RobotIL.constants import DT
# from utils.utils import set_seed
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
        self.pred_contact = None  # Initialize pred_contact
        self._set_up()

    def _set_up(self):
        """Sets up the inference process by loading the policy and initializing the environment."""
        # self._initialize_seed()
        self._load_policy()
        self._initialize_environment()
        self.temporal_agg = self.config.get("temporal_agg", False)
        self.query_frequency = 5
        if self.temporal_agg and self.config["policy_class"] == "ACT":
            self.num_queries = self.config["policy_config"]["num_queries"]
            # num_queries default 100 for ACT

    # def _initialize_seed(self):
    #     """Sets the random seed for reproducibility."""
    #     set_seed(self.config["seed"])

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

    def _initialize_environment(self):
        """This method should be overridden by environment-specific subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
        
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
                self.all_actions, self.pred_contact = self.policy(qpos, curr_image)
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

        return raw_action, self.pred_contact, self.all_actions
    
   
        
    def rotation_6d_to_quaternion(self, rot_6d):
        v1 = rot_6d[:3]  # First column
        v2 = rot_6d[3:]  # Second column

        # Compute the third column by taking the cross product of m1 and m2
        # This ensures that the third column is orthogonal to the first two
        v1 = v1 / np.linalg.norm(v1)

        # Step 2: Make the second vector orthogonal to the first
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)

        # Reconstruct the full 3x3 rotation matrix
        rot_matrix = np.stack([v1, v2, v3], axis=-1)  # 3x3 matrix

        # Convert the rotation matrix back to a quaternion
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # Quaternion in [x, y, z, w] format

        # Return quaternion in [w, x, y, z] format for consistency
        return quat  
        

    def action_process(self, action):
        predicted_pos = action[:3]
        predicted_rot_6d = action[3:-1]
        predicted_gripper = action[-1]
        predicted_gripper = np.array([predicted_gripper])
        predicted_quat = self.rotation_6d_to_quaternion(predicted_rot_6d)

        norm = np.linalg.norm(predicted_quat)
        if norm == 0:
            raise ValueError("The quaternion has zero magnitude and cannot be normalized.")
        predicted_quat = predicted_quat / norm
        action_all = np.concatenate([predicted_pos, predicted_quat,predicted_gripper], axis=-1)
        return action_all
    
    
    def _get_data(self, t):
        # qpos_history = torch.zeros((1, max_timesteps, self.config["state_dim"])).cuda()
        # rgb_images = []

        # for t in range(max_timesteps):
        #     rgb_list, joints = self.env.get_obs_joint()
        #     qpos_numpy = np.array(joints)
        #     qpos = self._pre_process(qpos_numpy)
        #     qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        #     qpos_history[:, t] = qpos

        #     curr_image = np.array(rgb_list)
        #     curr_image = torch.from_numpy(curr_image) / 255.0
        #     curr_image = (
        #         curr_image.permute(0, 3, 1, 2).view((1, -1, 3, 480, 640)).cuda()
        #     )
        #     rgb_images.append(curr_image)

        # return qpos_history, rgb_images
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def _run(self, qpos, rgb_images, t, all_time_actions=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    
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
        print(f"num_queries is {num_queries}")
        start_time = time.time()

        self.env.seed(0)

        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, self.config["policy_config"]["state_dim"]]
            ).cuda()
        # qpos_history = torch.zeros((1, max_timesteps, self.config["policy_config"]["state_dim"])).cuda()
        with torch.no_grad():
            print(f"temporal_agg is {self.temporal_agg}")
            for t in range(max_timesteps):
                time.sleep(0.01)
                qpos, rgb_images, door_pose, pcd_from_mesh = self._get_data(t)
                # self.test_by_collect(t)
                # Run the collected data through the policy
                self._run(
                    qpos,
                    rgb_images,
                    t,
                    all_time_actions if temporal_agg else None,
                    door_pose, pcd_from_mesh

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