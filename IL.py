import time

# import jax

# Global flag to set a specific platform, must be used at startup.
# jax.config.update('jax_platform_name', 'cpu')

import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange

from constants import DT, SIM_TASK_CONFIGS

# from core.env.fold3 import Fold3Env
from RobotIL.utils.utils import load_data  # data functions
from RobotIL.utils.utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from RobotIL.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

import os
import torch
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    num_epochs = args["num_epochs"]

    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 20
    episode_len = 300
    lr_backbone = 1e-5
    backbone = args["visual_encoder"]  # ["dinov2", "resnet18"]
    variant = args["variant"]  # ["base", "large"]
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "variant": variant,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "variant": variant,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    elif policy_class == "Diffusion":
        policy_config = {
            "lr": args["lr"],
            "camera_names": camera_names,
            "backbone": backbone,
            "variant": variant,
            "action_dim": 20,
            "observation_horizon": 1,
            "action_horizon": 8,
            "prediction_horizon": args["chunk_size"],
            "num_queries": args["chunk_size"],
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "starting_point_control": args["starting_point_control"],
        "camera_names": camera_names,
        "batch_size": args["batch_size"],
        "predict_value": args["predict_value"],
    }

    if is_eval:
        ckpt_names = ["policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        return

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    world_size = torch.cuda.device_count()
    mp.spawn(train_bc, args=(world_size, config), nprocs=world_size, join=True)
    # train_bc(0, world_size, config)


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    from core.real_robot.fetch import FetchEnv

    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    starting_point_control = config["starting_point_control"]
    onscreen_cam = "angle"
    predict_value = config["predict_value"]

    # Load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda:0"))
        loading_status = policy.load_state_dict(checkpoint)
        print(loading_status)

    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    def pre_process(s_qpos):
        return s_qpos

    def post_process(a):
        return a

    env_max_reward = 1
    env = FetchEnv(task_name, predict_value=predict_value)
    if starting_point_control:
        env.starting_point_control()

    query_frequency = 5
    if temporal_agg:
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        env.seed(rollout_id)
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
        time_start = time.time()

        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                rgb_list, joints = env.get_obs_joint()
                image_list.append(rgb_list)
                qpos_numpy = np.array(joints)
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                curr_image = np.array(rgb_list)
                curr_image = torch.from_numpy(curr_image) / 255.0
                curr_image = (
                    curr_image.permute(0, 3, 1, 2).view((1, -1, 3, 480, 640)).cuda()
                )  # (480, 640, 3) -> (3, 480, 640)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.1
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                elif config["policy_class"] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = raw_action
                target_qpos = action

                if predict_value == "ee_pos_ori":
                    env.ee_space_control(raw_action)
                else:
                    env.step(raw_action)

                print(t, target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(0)

            plt.close()

        print(f"Rollout {rollout_id} took {time.time() - time_start} seconds")

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards is not None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}"
        )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n"

    print(summary_str)

    # Save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(rank, world_size, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    train_dataloader, val_dataloader, stats, _ = load_data(
        config["batch_size"],
        config["batch_size"],
        world_size,
        rank,
        config["task_name"],
        config["predict_value"],
    )
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Initialize the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    policy = make_policy(policy_class, policy_config)

    # Set the device according to the local rank (GPU id)
    device = torch.device(f"cuda:{rank}")
    policy.to(device)

    ckpt_name = "policy_best.ckpt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(f"loading {ckpt_name}", loading_status)

    # Convert the policy model to DDP
    policy = DDP(
        policy, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    policy.state_dict = policy.module.state_dict
    policy.configure_optimizers = policy.module.configure_optimizers

    optimizer = make_optimizer(policy_class, policy)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler("cuda")

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        # Evaluation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"].mean()
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.mean().item():.3f} "
        print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            with autocast("cuda", dtype=torch.float16):  # Enable mixed precision
                forward_dict = forward_pass(data, policy)
                loss = forward_dict["loss"].mean()

            # Scale loss and perform backward pass
            scaler.scale(loss).backward()

            # Step the optimizer
            scaler.step(optimizer)

            # Update the scaler
            scaler.update()

            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"].mean()
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.mean().item():.3f} "
        print(summary_string)

        if epoch % 50 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # Clean up the distributed environment
    dist.destroy_process_group()


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].mean().item() for summary in train_history]
        val_values = [summary[key].mean().item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=False
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        default="ACT",
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=False
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", default=32
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", default=0)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", default=2000
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", default=1e-5)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", default=10
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="chunk_size",
        required=False,
        default=100,
    )
    parser.add_argument(
        "--hidden_dim",
        action="store",
        type=int,
        help="hidden_dim",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        default=3200,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--starting_point_control", action="store_true")
    parser.add_argument(
        "--predict_value", action="store", type=str, default="ee_pos_ori"
    )
    parser.add_argument("--visual_encoder", action="store", type=str, default="dinov2")
    parser.add_argument("--variant", action="store", type=str, default="vits14")

    main(vars(parser.parse_args()))