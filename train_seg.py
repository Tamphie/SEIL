# segment_train.py (dedicated script for segmentation model only)
import os
import gc
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from RobotIL.utils.utils import load_data, compute_dict_mean, set_seed, detach_dict
from RobotIL.policy import PointNetSegModel
import torch.nn.functional as F

class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        set_seed(config["seed"])
        self.seg_ckpt_dir = os.path.join(config["ckpt_dir"], "seg_model")
        os.makedirs(self.seg_ckpt_dir, exist_ok=True)

    def make_seg_model(self):
        return PointNetSegModel(self.config["seg_config"])

    def make_optimizer(self, model):
        seg_config = self.config["seg_config"]
        lr = seg_config.get("learning_rate", 0.001)
        if seg_config.get("optimizer", "adam") == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        else:
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=seg_config.get("momentum", 0.9))

    def forward_pass(self, data, model, device):
        image_data, _, _, _, contact_data = data
        point_cloud = image_data.to(device)
        gt_labels = contact_data.to(device).squeeze().long()
        output = model(point_cloud)
        pred = output['seg_pred']
        pred_labels = output['seg_pred_labels']
        loss = F.cross_entropy(pred.transpose(1, 2), gt_labels)
        accuracy = (pred_labels == gt_labels).float().mean()
        return {
            "seg_loss": loss,
            "seg_accuracy": accuracy,
            "seg_pred": pred,
            "seg_pred_labels": pred_labels
        }
    def plot_segmentation_history(self, train_history, validation_history, num_epochs, seed):
        for key in train_history[0]:
            plot_path = os.path.join(self.seg_ckpt_dir, f"seg_train_val_{key}_seed_{seed}.png")
            import matplotlib.pyplot as plt
            plt.figure()
            train_values = [summary[key].float().mean().item() for summary in train_history]
            val_values = [summary[key].float().mean().item() for summary in validation_history]
            # plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train")
            plt.plot(range(len(train_history)), train_values, label="train")
            # plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label="validation")
            plt.plot(range(len(validation_history)), val_values, label="validation")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.legend()
            plt.title(f"Segmentation {key}")
            plt.savefig(plot_path)
        print(f"Saved segmentation plots to {self.seg_ckpt_dir}")

    def train_process(self, rank, world_size):
        config = self.config
        num_epochs = config["num_epochs"]
        seed = config["seed"]

        train_loader, val_loader, stats, _, _ = load_data(
            config["batch_size"], config["batch_size"], world_size, rank,
            config["task_name"], config["predict_value"], config["obs_type"]
        )

        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12355"
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        device = torch.device(f"cuda:{rank}")
        model = self.make_seg_model().to(device)

        seg_ckpt_path = os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt")
        if os.path.exists(seg_ckpt_path):
            model.load_state_dict(torch.load(seg_ckpt_path, map_location=device))

        # model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        model.state_dict = model.module.state_dict

        optimizer = self.make_optimizer(model)
        scaler = torch.cuda.amp.GradScaler()

        train_history, val_history = [], []
        min_val_loss = float("inf")
        best_ckpt_info = None

        for epoch in tqdm(range(num_epochs), desc=f"Rank {rank}"):
            if rank == 0:
                model.eval()
                with torch.inference_mode():
                    summaries = [self.forward_pass(batch, model, device) for batch in val_loader]
                    val_summary = compute_dict_mean(summaries)
                    val_loss = val_summary["seg_loss"].mean()
                    val_history.append(val_summary)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_ckpt_info = (epoch, deepcopy(model.state_dict()))

            model.train()
            optimizer.zero_grad()
            for batch in train_loader:
                with torch.cuda.amp.autocast():
                    output = self.forward_pass(batch, model, device)
                    loss = output["seg_loss"].mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_history.append(detach_dict(output))
                gc.collect()
                torch.cuda.empty_cache()

            if rank == 0 and epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join(self.seg_ckpt_dir, f"seg_model_epoch_{epoch}_seed_{seed}.ckpt"))

        if rank == 0 and best_ckpt_info:
            epoch, best_state = best_ckpt_info
            torch.save(best_state, os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt"))
        dist.destroy_process_group()
        if rank == 0:
            self.plot_segmentation_history(train_history, val_history, num_epochs, seed)


    def train(self):
        mp.spawn(self.train_process, args=(torch.cuda.device_count(),), nprocs=torch.cuda.device_count())

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        action="store",
        type=str,
        help="Checkpoint directory",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="Task name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="Batch size", default=32
    )
    parser.add_argument(
        "--seed", action="store", type=int, help="Random seed", default=0
    )
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="Number of epochs", default=2000
    )
    parser.add_argument(
        "--lr", action="store", type=float, help="Learning rate", default=1e-5
    )
    parser.add_argument(
        "--train_segmentation",
        action="store_true",
        help="Whether to train the segmentation model",
    )
    parser.add_argument(
        "--joint_training",
        action="store_true",
        help="Whether to jointly train policy and segmentation",
    )
    parser.add_argument(
        "--pointnet_dir",
        action="store",
        type=str,
        default="pointnet2",
        help="Directory containing PointNet++ code",
    )
    parser.add_argument(
        "--seg_lr",
        action="store",
        type=float,
        default=0.001,
        help="Learning rate for segmentation model",
    )
    parser.add_argument(
        "--seg_num_point",
        action="store",
        type=int,
        default=2048,
        help="Number of points for segmentation model",
    )
    parser.add_argument(
        "--seg_num_classes",
        action="store",
        type=int,
        default=2,
        help="Number of segmentation classes (default: 2 for contact/non-contact)",
    )
    parser.add_argument(
        "--seg_weight",
        action="store",
        type=float,
        default=1.0,
        help="Weight for segmentation loss during joint training",
    )
    parser.add_argument(
        "--seg_optimizer",
        action="store",
        type=str,
        default="adam",
        help="Optimizer for segmentation model (adam or momentum)",
    )
    parser.add_argument(
        "--predict_value",
        action="store",
        type=str,
        default="ee_pos_ori",
        help="Predict value type",
    )
    parser.add_argument(
        "--obs_type",
        action="store",
        type=str,
        default="rgbd",
        help="rgbd or pcd",
    )
    return vars(parser.parse_args())
# Example usage
if __name__ == "__main__":

    args = parse_arguments()
    seg_config = {
        "learning_rate": args["seg_lr"],
        "num_point": args["seg_num_point"],
        "num_classes": args["seg_num_classes"],
        "optimizer": args["seg_optimizer"],
        "momentum": 0.9,
    }

    config = {
        "num_epochs": args["num_epochs"],
        "ckpt_dir": args["ckpt_dir"],
        "lr": args["lr"],
        "task_name": args["task_name"],
        "seed": args["seed"],
        "batch_size": args["batch_size"],
        "predict_value": args["predict_value"],
        "obs_type": args["obs_type"],
        "seg_config": seg_config
    }

    trainer = SegmentationTrainer(config)
    trainer.train()
