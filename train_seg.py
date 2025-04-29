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
    def downsample(self, data, num_points):
        point_cloud, _, _, _, contact_data = data # B, N, C = point_cloud.shape = B, 10000,3 ; contact.shape = B, 1, 10000
        B, N, C = point_cloud.shape
        if N > num_points:
            idx = np.random.choice(N, num_points, replace=False)
        else:
            idx = np.random.choice(N, num_points, replace=True)
        idx = torch.tensor(idx, dtype=torch.long)
        point_cloud = point_cloud[:, idx , :] #([16, 1024, 3])
        contact_data = contact_data[:,:, idx] #([16, 1, 1024])
        return point_cloud, contact_data, idx
    
    def forward_pass(self, data, model, device):
        num_points = self.config["seg_config"]["num_point"]
        point_cloud, contact_data, _ = self.downsample(data, num_points)
        point_cloud = point_cloud.to(device)
        gt_labels = contact_data.to(device).squeeze().long()
        output = model(point_cloud,gt_labels)
        # pred = output['seg_pred']
        # pred_labels = output['seg_pred_labels']
        # loss = F.cross_entropy(pred.transpose(1, 2), gt_labels)
        # accuracy = (pred_labels == gt_labels).float().mean()
        return output
    # {
    #         "seg_loss": loss,
    #         "seg_accuracy": accuracy,
    #         "seg_pred": pred,
    #         "seg_pred_labels": pred_labels
    #     }
    def plot_segmentation_history(self, train_history, validation_history, num_epochs, seed):
        for key in train_history[0]:
            plot_path = os.path.join(self.seg_ckpt_dir, f"seg_train_val_{key}_seed_{seed}.png")
            import matplotlib.pyplot as plt
            plt.figure()
            train_values = [summary[key].float().mean().item() for summary in train_history]
            val_values = [summary[key].float().mean().item() for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train")
            # plt.plot(range(len(train_history)), train_values, label="train")
            plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label="validation")
            # plt.plot(range(len(validation_history)), val_values, label="validation")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.legend()
            plt.title(f"Segmentation {key}")
            plt.savefig(plot_path)
        print(f"Saved segmentation plots to {self.seg_ckpt_dir}")

    def train_process_1(self, rank=0, world_size=1):
        config = self.config
        num_epochs = config["num_epochs"]
        seed = config["seed"]

        train_loader, val_loader, stats, _, _ = load_data(
            config["batch_size"], config["batch_size"], world_size, rank,
            config["task_name"], config["predict_value"], config["obs_type"]
        )

        # os.environ["MASTER_ADDR"] = "localhost"
        # # os.environ["MASTER_PORT"] = "12355"
        # os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        # os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

        # dist.init_process_group("nccl", rank=rank, world_size=world_size)

        device = torch.device("cuda")
        model = self.make_seg_model().to(device)

        seg_ckpt_path = os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt")
        if os.path.exists(seg_ckpt_path):
            model.load_state_dict(torch.load(seg_ckpt_path, map_location=device))

        # model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # model.state_dict = model.module.state_dict

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
                    torch.cuda.empty_cache()
                    gc.collect()
                    val_summary = compute_dict_mean(summaries)
                    val_loss = val_summary["seg_loss"].mean()
                    val_history.append(val_summary)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_ckpt_info = (epoch, deepcopy(model.state_dict()))
                gc.collect()
                torch.cuda.empty_cache()
                
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
                

            if rank == 0 and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(self.seg_ckpt_dir, f"seg_model_epoch_{epoch}_seed_{seed}.ckpt"))
                self.plot_segmentation_history(train_history, val_history, num_epochs, seed)

        if rank == 0 and best_ckpt_info:
            epoch, best_state = best_ckpt_info
            torch.save(best_state, os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt"))
        # dist.destroy_process_group()

    def train_process(self, rank=0, world_size=1):
        config = self.config
        num_epochs = config["num_epochs"]
        seed = config["seed"]

        train_loader, val_loader, stats, _, _ = load_data(
            config["batch_size"], config["batch_size"], world_size, rank,
            config["task_name"], config["predict_value"], config["obs_type"]
        )

        device = torch.device("cuda")
        model = self.make_seg_model().to(device)

        seg_ckpt_path = os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt")
        if os.path.exists(seg_ckpt_path):
            model.load_state_dict(torch.load(seg_ckpt_path, map_location=device))

        optimizer = self.make_optimizer(model)
        scaler = torch.cuda.amp.GradScaler()

        # Use lists only for storing recent history
        train_history_window = []
        val_history_window = []
        min_val_loss = float("inf")
        best_ckpt_info = None

        print(f"Starting training with batch size {config['batch_size']}, points {config['seg_config']['num_point']}")
        
        for epoch in tqdm(range(num_epochs), desc=f"Rank {rank}"):
            # Validation step
            if rank == 0:
                model.eval()
                val_summaries = []
                with torch.inference_mode():
                    for batch_idx, batch in enumerate(val_loader):
                        with torch.cuda.amp.autocast():
                            output = self.forward_pass(batch, model, device)
                        val_summaries.append(detach_dict(output))
                        # Clean memory after each batch
                        del batch, output
                        if batch_idx % 5 == 0:  # Clean every 5 batches
                            torch.cuda.empty_cache()
                    
                    val_summary = compute_dict_mean(val_summaries)
                    val_loss = val_summary["seg_loss"].mean()
                    val_history_window.append(val_summary)
                    
                    # Keep only recent validation history
                    if len(val_history_window) > 10:
                        val_history_window.pop(0)
                    
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_ckpt_info = (epoch, deepcopy(model.state_dict()))
                        print(f"New best model at epoch {epoch}, val_loss: {val_loss:.4f}")
                    
                del val_summaries
                torch.cuda.empty_cache()
                    
            # Training step
            model.train()
            train_summaries = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Clear memory and gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    output = self.forward_pass(batch, model, device)
                    loss = output["seg_loss"].mean()
                
                # Backward and optimize with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Store batch results 
                train_summaries.append(detach_dict(output))
                
                # Clean memory after each batch
                del batch, output, loss
                torch.cuda.empty_cache()
                
                # Print batch progress periodically
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            # Compute epoch stats and update history
            train_summary = compute_dict_mean(train_summaries)
            train_history_window.append(train_summary)
            
            # Keep only recent training history
            if len(train_history_window) > 10:
                train_history_window.pop(0)
                
            # Print epoch summary
            print(f"Epoch {epoch}: train_loss={train_summary['seg_loss'].mean():.4f}, train_acc={train_summary['seg_accuracy'].mean():.4f}")
            
            # Clean memory after each epoch
            del train_summaries
            gc.collect()
            torch.cuda.empty_cache()

            # Save checkpoint periodically
            if rank == 0 and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(self.seg_ckpt_dir, f"seg_model_epoch_{epoch}_seed_{seed}.ckpt"))
                self.plot_segmentation_history(train_history_window, val_history_window, len(train_history_window), seed)

        # Save best model at the end
        if rank == 0 and best_ckpt_info:
            epoch, best_state = best_ckpt_info
            torch.save(best_state, os.path.join(self.seg_ckpt_dir, "seg_model_best.ckpt"))
            print(f"Training complete. Best model saved from epoch {epoch}")
    def train(self):
        # mp.spawn(self.train_process, args=(torch.cuda.device_count(),), nprocs=torch.cuda.device_count())
        self.train_process()

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
