# RLBench: Robot Learning Benchmark [![Unit Tests](https://github.com/stepjam/RLBench/workflows/Unit%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Task Tests](https://github.com/stepjam/RLBench/workflows/Task%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Discord](https://img.shields.io/discord/694945190867370155.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/DXPCjmd)

![task grid image missing](readme_files/task_grid.png)

**RLBench** is an ambitious large-scale benchmark and learning environment 
designed to facilitate research in a number of vision-guided manipulation
research areas, including: reinforcement learning, imitation learning,
multi-task learning, geometric computer vision, and in particular, 
few-shot learning. [Click here for website and paper.](https://sites.google.com/corp/view/rlbench)

**Contents:**
- [Announcements](#announcements)
- [Install](#install)
- [Running Headless](#running-headless)
- [Getting Started](#getting-started)
    - [Few-Shot Learning and Meta Learning](#few-shot-learning-and-meta-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Sim-to-Real](#sim-to-real)
    - [Imitation Learning](#imitation-learning)
    - [Multi-Task Learning](#multi-task-learning)
    - [RLBench Gym](#rlbench-gym)
    - [Swapping Arms](#swapping-arms)
- [Tasks](#tasks)
- [Task Building](#task-building)
- [Gotchas!](#gotchas)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

# Original RlBench Content
## Install

RLBench is built around CoppeliaSim v4.1.0 and [PyRep](https://github.com/stepjam/PyRep).

First, install CoppeliaSim:

```bash
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

To install the RLBench python package:

```bash
pip install git+https://github.com/stepjam/RLBench.git
```

And that's it!

## Running Headless

If you are running on a machine without display (i.e. Cloud VMs, compute clusters),
you can refer to the following guide to run RLBench headlessly with rendering.

### Initial setup

First, configure your X config. This should only be done once to set up.

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf
```

Leave out `--use-display-device=None` if the GPU is headless, i.e. if it has no display outputs.

### Running X

Then, whenever you want to run RLBench, spin up X.

```bash
# nohup and disown is important for the X server to keep running in the background
sudo nohup X :99 & disown
```

Test if your display works using glxgears.

```bash
DISPLAY=:99 glxgears
```

If you have multiple GPUs, you can select your GPU by doing the following.

```bash
DISPLAY=:99.<gpu_id> glxgears
```

### Running X without sudo

To spin up X with non-sudo users, edit file '/etc/X11/Xwrapper.config' and replace line:

```
allowed_users=console
```
with lines:
```
allowed_users=anybody
needs_root_rights=yes
```
If the file does not exist already, you can create it.

## Getting Started

The benchmark places particular emphasis on few-shot learning and meta learning 
due to breadth of tasks available, though it can be used in numerous ways. Before using RLBench, 
checkout the [Gotchas](#gotchas) section.

### Imitation Learning

```python
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget

# To use 'saved' demos, set the path below
DATASET = 'PATH/TO/YOUR/DATASET'

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode, DATASET)
env.launch()

task = env.get_task(ReachTarget)

demos = task.get_demos(2)  # -> List[List[Observation]]
demos = np.array(demos).flatten()

batch = np.random.choice(demos, replace=False)
batch_images = [obs.left_shoulder_rgb for obs in batch]
predicted_actions = predict_action(batch_images)
ground_truth_actions = [obs.joint_velocities for obs in batch]
loss = behaviour_cloning_loss(ground_truth_actions, predicted_actions)

```

A full example can be seen in [examples/imitation_learning.py](examples/imitation_learning.py).


### Swapping Arms

The default Franka Panda Arm _can_ be swapped out for another. This can be
useful for those who have custom tasks or want to perform sim-to-real 
experiments on the tasks. However, if you swap out the arm, then we can't 
guarantee that the task will be solvable.
For example, the Mico arm has a very small workspace in comparison to the
Franka.

**For benchmarking, the arm should remain as the Franka Panda.**

Currently supported arms:

- Franka Panda arm with Franka gripper `(franka)`
- Mico arm with Mico gripper `(mico)`
- Jaco arm with 3-finger Jaco gripper `(jaco)`
- Sawyer arm with Baxter gripper `(sawyer)`
- UR5 arm with Robotiq 85 gripper `(ur5)`

You can then swap out the arm using `robot_configuration`:

```python
env = Environment(action_mode=action_mode, robot_setup='sawyer')
```

A full example (using the Sawyer) can be seen in [examples/swap_arm.py](examples/swap_arm.py).

_Don't see the arm that you want to use?_ Your first step is to make sure it is
in PyRep, and if not, then you can follow the instructions for importing new
arm on the PyRep GitHub page. After that, feel free to open an issue and 
we can being it in to RLBench for you.

## Tasks

To see a full list of all tasks, [see here](rlbench/tasks).


## Citation

```
@article{james2019rlbench,
  title={RLBench: The Robot Learning Benchmark \& Learning Environment},
  author={James, Stephen and Ma, Zicong and Rovick Arrojo, David and Davison, Andrew J.},
  journal={IEEE Robotics and Automation Letters},
  year={2020}
}
```
# SELF Content
## Record demos
Set up
```bash
# In terminal 1
pip install --user virtualenv
virtualenv SEIL
source SEIL/bin/activate
pip install --upgrade pip setuptools


export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget --no-check-certificate https://downloads.coppeliaroboticsUbuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_UOOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

pip install git+https://github.com/stepjam/RLBench.git

pip install gymnasium 
pip install testresources

pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
# Install the headless version of OpenCV
pip install opencv-python-headless
pip install PyQt5
```
This is how to build task:
You can configure the saved_path and task in the scripts
```bash
cd SEIL/SEIL
python3 tools/task_builder.py

```
This is how to collect data
```bash
#Open MobaXterm
source ENV/bin/activate
export ..
pip install opencv-python-headless
cd SEIL/
bash scripts/generate_dataset_IL.sh
#variations have to be larger than 0

```
To see the collected image in MobaXterm
``` bash
eog /home/tongmiao/SEIL/SEIL/data/open_door/episode_0/right_shoulder_rgb/0.png 


```
To train the data:
```bash
git clone https://github.com/RobotIL-rls/RobotIL.git --recursive
git clone https://github.com/RobotIL-rls/robomimic.git
cd RobotIL/
pip install -e .
pip install -e robomimic
pip install opencv-python
bash scripts/train_policy.sh

```
To infer:
```bash
source ENV/bin/activate
export ..
pip install opencv-python-headless
cd SEIL/
bash scripts/inference.sh
