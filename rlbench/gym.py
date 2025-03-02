from typing import Union
import re
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode


from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

def convert_dtype_to_float32_if_float(dtype):
    if issubclass(dtype.type, np.floating):
        return np.float32
    return dtype

class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, task_class, observation_mode='vision',
                 render_mode: Union[None, str] = None, action_mode=None):
        self.task_class = task_class
        self.observation_mode = observation_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        self.obs_config = obs_config
        if action_mode is None:
            action_mode = JointPositionActionMode()
        self.action_mode = action_mode

        self.rlbench_env = Environment(
            action_mode=self.action_mode,
            obs_config=self.obs_config,
            headless=False,
        )
        self.rlbench_env.launch()
        self.rlbench_task_env = self.rlbench_env.get_task(self.task_class)
        if render_mode is not None:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([640, 360])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)
        _, obs = self.rlbench_task_env.reset()

        gym_obs = self._extract_obs(obs)
        self.observation_space = {}
        for key, value in gym_obs.items():
            if "rgb" in key:
                self.observation_space[key] = spaces.Box(
                    low=0, high=255, shape=value.shape, dtype=value.dtype)
            else:
                self.observation_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
        self.observation_space = spaces.Dict(self.observation_space)

        action_low, action_high = action_mode.action_bounds()
        print(f"self.rlbench_env.action_shape:{self.rlbench_env.action_shape}")
        self.action_space = spaces.Box(
            low=np.float32(action_low), high=np.float32(action_high), shape=self.rlbench_env.action_shape, dtype=np.float32)
        # if isinstance(self.action_space, spaces.Box):
        #     print(f"🔹 Action space: {self.action_space.low} to {self.action_space.high}")
        # else:
        #     print("type of space action",type(self.action_space))
    def _extract_obs(self, rlbench_obs):
        print("????Available states in rlbench_obs:", dir(rlbench_obs))
        gym_obs = {} 
        for state_name in ["joint_velocities", "joint_positions", "joint_forces", "gripper_open", "gripper_pose", "gripper_joint_positions", "gripper_touch_forces", "task_low_dim_state", "pcd_from_mesh", "dist_data","front_point_cloud"]:
            state_data = getattr(rlbench_obs, state_name)
            print(f"\n🔹 Processing state: {state_name}")
            # print(f"   - Original data: {state_data}")
            # print(f"   - Original type: {type(state_data)}")
            if state_data is not None:
                print("state data is not none")
                if isinstance(state_data, np.ndarray) and state_data.dtype.type is np.str_:
                    # Convert only valid numbers, silently skip strings
                    state_data = np.array([float(val) for val in state_data 
                                        if re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", val)], dtype=np.float32)
                else:
                    state_data = np.float32(state_data)  # Convert normally if already numeric
                
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs[state_name] = state_data
                
        if self.observation_mode == 'vision':
            gym_obs.update({
                "left_shoulder_rgb": rlbench_obs.left_shoulder_rgb,
                "right_shoulder_rgb": rlbench_obs.right_shoulder_rgb,
                "wrist_rgb": rlbench_obs.wrist_rgb,
                "front_rgb": rlbench_obs.front_rgb,
            })
        return gym_obs

    def render(self):
        if self.render_mode == 'rgb_array':
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: Remove this and use seed from super()
        np.random.seed(seed=seed)
        reset_to_demo = None
        if options is not None:
            # TODO: Write test for this
            reset_to_demo = options.get("reset_to_demo", None)

        if reset_to_demo is None:
            descriptions, obs = self.rlbench_task_env.reset()
        else:
            descriptions, obs = self.rlbench_task_env.reset(reset_to_demo=reset_to_demo)
        return self._extract_obs(obs), {"text_descriptions": descriptions}

    def step(self, action):
        obs, reward, terminated = self.rlbench_task_env.step(action)
        return self._extract_obs(obs), reward, terminated, False, {}

    def close(self) -> None:
        self.rlbench_env.shutdown()
        

