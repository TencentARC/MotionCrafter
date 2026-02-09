# Base Trainer Module
# Provides abstract base class for training motion generation models
# with distributed training support and checkpoint management

import logging
from dataclasses import dataclass
from datetime import timedelta

import gin
import math
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
)

from datasets import DatasetConfig

# Check minimum diffusers version
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def gin_config_to_readable_dictionary(gin_config: dict) -> dict:
    # Convert gin configuration to readable dictionary for logging
    # Useful for logging to W&B or other tracking services
    # Args:
    #   gin_config: The gin config dictionary from gin.config._OPERATIVE_CONFIG
    # Returns:
    #   Parsed and cleaned configuration dictionary
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data


class BaseTrainer:
    # Base trainer class for distributed training of motion generation models
    # Handles distributed setup, data loading, logging, and checkpointing
    
    def __init__(self, args: dataclass):
        # Initialize trainer with configuration
        # Sets up distributed training, logging, and data pipelines
        self.config = args

        # Setup W&B if requested
        if self.config.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Install wandb if using it for logging."
                )
            import wandb

            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = str(self.config.logging_path)

        # Handle non-EMA revision deprecation
        if self.config.non_ema_revision is not None:
            deprecate(
                "non_ema_revision!=None",
                "0.15.0",
                message=(
                    "Use `--variant=non_ema` instead of downloading from "
                    "revision branches."
                ),
            )

        # Initialize distributed training with Accelerate
        # Configure gradient accumulation, mixed precision, and logging
        self.accelerator = Accelerator(
            gradient_accumulation_steps=(
                self.config.gradient_accumulation_steps
            ),
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.report_to,
            project_config=ProjectConfiguration(
                project_dir=str(self.config.exp_path),
                logging_dir=str(self.config.logging_path),
            ),
            kwargs_handlers=[
                InitProcessGroupKwargs(
                    timeout=timedelta(seconds=5400)
                ),  # 1.5 hours
            ],
        )
        # Setup random generators for reproducibility
        self.batch_generator = (
            torch.Generator(device='cpu').manual_seed(
                self.config.seed + self.accelerator.process_index
            )
        )
        self.generator = torch.Generator(
            device=self.accelerator.device
        ).manual_seed(self.config.seed)

        # Configure logging for debugging
        logging.basicConfig(
            filename=self.config.logging_path / "log.txt",
            filemode="w",
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            ),
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        # Adjust log verbosity based on process rank
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Set training seed for reproducibility
        if self.config.seed is not None:
            set_seed(
                self.config.seed + self.accelerator.process_index
            )

        # Determine precision dtype for models
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Enable TF32 for faster training on Ampere GPUs if requested
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Build dataset and dataloader
        dataset_cfg = DatasetConfig()
        train_dataset = dataset_cfg.build()
        
        # Select sampler based on configuration
        if dataset_cfg.batch_sampler == 'random':
            sampler = RandomSampler(train_dataset)
        elif dataset_cfg.batch_sampler == 'sequential':
            sampler = SequentialSampler(train_dataset)
        else:
            raise NotImplementedError
        
        # Create dataloader with specified batch size and workers
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.per_gpu_batch_size,
            num_workers=args.num_workers,
            collate_fn=None,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Calculate training steps for scheduler and logging
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader)
            / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = (
                args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        # Initialize trackers and save configuration
        if self.accelerator.is_main_process:
            all_cfg = gin.config._OPERATIVE_CONFIG
            all_cfg = gin_config_to_readable_dictionary(all_cfg)
            self.accelerator.init_trackers(
                "PointMapDiffusion",
                config=all_cfg,
                init_kwargs={
                    "wandb": {"name": args.exp_name, "mode": "offline"}
                },
            )
            if args.report_to == "wandb":
                from wandb_osh.hooks import TriggerWandbSyncHook

                self.trigger_sync = TriggerWandbSyncHook()
            
            # Save gin configuration
            config_save_path = (
                self.config.exp_path / "configs"
                / self.accelerator.get_tracker("wandb").run.id
            )
            config_save_path.mkdir(parents=True, exist_ok=True)
            conf = gin.operative_config_str()
            with open(self.config.exp_path / "config.gin", "w") as f:
                f.write(conf)


    def resume(self, checkpoint_path: Path = None):
        # Resume training from checkpoint or most recent one
        if checkpoint_path is None:
            # Find most recent checkpoint
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = dirs[-1] if len(dirs) > 0 else None

        if checkpoint_path is None:
            self.accelerator.print(
                f"Checkpoint '{checkpoint_path}' not found. "
                "Starting new training."
            )
        else:
            self.accelerator.print(
                f"Resuming from checkpoint {checkpoint_path}"
            )
            self.accelerator.load_state(str(checkpoint_path))
            global_step = int(str(checkpoint_path).split("-")[1])
            return global_step

    def save_checkpoint(self, global_step):
        # Save checkpoint and manage checkpoint limit
        # Remove old checkpoints if limit exceeded
        if self.config.checkpoints_total_limit is not None:
            checkpoints = list(self.config.exp_path.glob("checkpoint-*"))
            checkpoints = sorted(
                checkpoints, key=lambda x: int(x.name.split("-")[1])
            )

            # Keep checkpoints_total_limit - 1 checkpoints before saving
            if len(checkpoints) >= self.config.checkpoints_total_limit:
                num_to_remove = (
                    len(checkpoints)
                    - self.config.checkpoints_total_limit
                    + 1
                )
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints exist, "
                    f"removing {len(removing_checkpoints)}"
                )
                logger.info(
                    f"removing checkpoints: {removing_checkpoints}"
                )

                for removing_checkpoint in removing_checkpoints:
                    try:
                        shutil.rmtree(removing_checkpoint)
                    except Exception as e:
                        logger.info(
                            f"Error removing {removing_checkpoint}: {e}"
                        )

        # Save current checkpoint
        save_path = self.config.exp_path / f"checkpoint-{global_step}"
        save_path.mkdir(parents=True, exist_ok=True)

        self.accelerator.save_state(str(save_path))
        logger.info(f"Saved state to {save_path}")

    def train(self):
        # Abstract method: main training loop
        # Implemented by subclasses
        raise NotImplementedError

    def train_iter(
        self,
        batch,
        **kwargs,
    ) -> torch.Tensor:
        # Abstract method: single training iteration
        # Args:
        #   batch: Batch of training data
        #   kwargs: Additional arguments
        # Returns:
        #   Loss tensor for backpropagation
        raise NotImplementedError

    def validate(
        self,
        **kwargs,
    ):
        # Abstract method: validation step
        # Args:
        #   kwargs: Additional arguments for validation
        raise NotImplementedError
