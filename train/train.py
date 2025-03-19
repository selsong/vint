import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

"""
IMPORT YOUR MODEL HERE
"""
import sys
sys.path.append('/bigdata/selina/vint_release')

from vint_train.models.gnm import GNM
from vint_train.models.vint import ViNT
from vint_train.models import vint
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    load_model,
    count_parameters,
)

# Monkey patch the train_eval_loop function to fix wandb logging
import types
from vint_train.training import train_eval_loop as tel_module

# Create a backup of the original function
original_train_eval_loop = tel_module.train_eval_loop

# Define our patched version
def patched_train_eval_loop(*args, **kwargs):
    # Get the use_wandb parameter
    use_wandb = kwargs.get('use_wandb', False)
    
    # Modify the kwargs to disable wandb if needed
    if use_wandb and not wandb.run:
        print("WARNING: wandb.init() has not been called, disabling wandb logging")
        kwargs['use_wandb'] = False
    
    # Call the original function
    return original_train_eval_loop(*args, **kwargs)

# Apply the monkey patch
tel_module.train_eval_loop = patched_train_eval_loop


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    # Process only one dataset/video
    dataset_processed = False
    
    for dataset_name in config["datasets"]:
        if dataset_processed:
            continue  # Skip after processing one dataset
            
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                try:
                    # Create the dataset normally
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    
                    # Limit to just a few samples from one video
                    # Use a Subset to limit the dataset to a smaller number of samples
                    single_video_indices = list(range(min(100, len(dataset))))  # First 100 frames or less
                    limited_dataset = Subset(dataset, single_video_indices)
                    
                    print(f"Original dataset size: {len(dataset)}")
                    print(f"Limited dataset size: {len(limited_dataset)}")
                    
                    if data_split_type == "train":
                        train_dataset.append(limited_dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = limited_dataset
                    
                    dataset_processed = True
                    break  # Process only one split
                except Exception as e:
                    print(f"Error loading dataset {dataset_name}, {data_split_type}: {str(e)}")
                    continue
        
        if dataset_processed:
            break  # Break after processing one dataset

    if not dataset_processed:
        print("WARNING: No dataset was processed. Check your dataset configurations.")
        return

    print(f"Processing only data from dataset: {dataset_name}")

    # combine all the datasets from different robots (should be just 1 now)
    if not train_dataset:
        print("ERROR: No training dataset found. Exiting.")
        return
        
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    else:
        raise ValueError(f"Model {config['model_type']} not supported")

    count_parameters(model)  # print number of parameters

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        try:
            load_project_folder = os.path.join("logs", config["load_run"])
            print("Loading model from ", load_project_folder)
            latest_path = os.path.join(load_project_folder, "latest.pth")
            latest_checkpoint = torch.load(latest_path, map_location=device)
            load_model(model, latest_checkpoint)
            current_epoch = latest_checkpoint["epoch"] + 1
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {str(e)}")

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config and 'latest_checkpoint' in locals():  # load optimizer and scheduler after data parallel
        try:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
            if scheduler is not None and "scheduler" in latest_checkpoint:
                scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
        except Exception as e:
            print(f"Warning: Could not load optimizer/scheduler states: {str(e)}")

    print("Starting train_eval_loop...")
    train_eval_loop(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        normalized=config["normalize"],
        print_log_freq=config["print_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        learn_angle=config["learn_angle"],
        alpha=config["alpha"],
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
    )
    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    
    try:
        os.makedirs(config["project_folder"])
    except FileExistsError:
        print(f"WARNING: Directory {config['project_folder']} already exists. Using it anyway.")

    # Ensure wandb is properly initialized before first use
    if config["use_wandb"]:
        try:
            wandb.login()
            wandb.init(
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="gnmv2", # TODO: change this to your wandb entity
            )
            wandb.save(args.config, policy="now")  # save the config file
            wandb.run.name = config["run_name"]
            # update the wandb args with the training configurations
            if wandb.run:
                wandb.config.update(config)
        except Exception as e:
            print(f"WARNING: Failed to initialize wandb: {str(e)}")
            print("Disabling wandb logging")
            config["use_wandb"] = False

    print(config)
    main(config)
