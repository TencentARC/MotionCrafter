"""Trainer for world-map VAE-KL with point-map and scene-flow reconstruction losses."""

from typing import Dict
import os
from pathlib import Path
from dataclasses import dataclass

import gin
import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from packaging import version
from tqdm.auto import tqdm
from einops import rearrange
from kornia.utils import create_meshgrid
import numpy as np
import pynvml
from copy import deepcopy

from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from motioncrafter import *
from .base import BaseTrainer
from utils.losses import (
    weighted_mse_loss, weighted_l1_loss, weighted_normal_loss,
    multi_scale_weighted_depth_loss, distance_weighted_mse_loss
)
from evaluation.metrics import project_to_depth_map
from utils.img_utils import save_image_tensor
from utils.pcd_utils import save_point_cloud
from utils.checkers import check_isnan

# Check minimum diffusers version
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

@gin.configurable()
class WMapVAEKLTrainer(BaseTrainer):
    # Trainer for world map VAE with KL divergence
    # Handles point map and scene flow encoding/decoding
    
    def __init__(
        self,
        args: dataclass,
        lambda_identity=1.0,
        lambda_wmap=1.0,
        lambda_sceneflow=1.0,
        lambda_deform_mask=0.1,
        lambda_valid_mask=0.1,
        lambda_l1_depth=1.0,
        lambda_patch_l1_depth=1.0,
        lambda_kl=0.0,
        lambda_normal=0.2,
        lambda_deform_normal=0.2,
    ):
        # Initialize trainer with loss weights
        # Args:
        #   args: Training configuration
        #   lambda_*: Loss weights for different components
        super().__init__(args)

        self.lambda_identity = lambda_identity
        self.lambda_wmap = lambda_wmap
        self.lambda_sceneflow = lambda_sceneflow
        self.lambda_deform_mask = lambda_deform_mask
        self.lambda_valid_mask = lambda_valid_mask
        self.lambda_l1_depth = lambda_l1_depth
        self.lambda_patch_l1_depth = lambda_patch_l1_depth
        self.lambda_kl = lambda_kl
        self.lambda_normal = lambda_normal
        self.lambda_deform_normal = lambda_deform_normal

        self._init_models()
        self._init_ckpt_hook()
        self._init_optimizers()
        pynvml.nvmlInit()

        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=(
                args.lr_warmup_steps * self.accelerator.num_processes
            ),
            num_training_steps=(
                args.max_train_steps * self.accelerator.num_processes
            ),
        )

        # Prepare everything with accelerator
        (
            self.geometry_motion_vae,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.geometry_motion_vae,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader
        )
        if args.use_ema:
            self.ema_vae.to(self.accelerator.device)

        # Initialize test case storage
        self.test_case = None
        self.test_case_gt = None

    def _init_models(self):
        # Initialize VAE models for RGB and point map encoding
        # Initialize VAE models for RGB and point map encoding
        assert self.config.pretrained_model_name_or_path is not None
        
        # Load temporal RGB VAE
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.config.pretrained_model_name_or_path,
            cache_dir=self.config.cache_dir,
            subfolder="vae",
            revision=self.config.revision,
            variant="fp16" if self.config.mixed_precision == 'fp16' else None
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # Load or initialize point map VAE
        if self.config.vae_type == "AutoencoderKL":
            self.geometry_motion_vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                cache_dir=self.config.cache_dir,
                subfolder="vae",
                revision=self.config.revision,
                variant="fp16" if (
                    self.config.mixed_precision == 'fp16'
                ) else None
            )
        else:
            if self.config.pretrained_vae is not None:
                self.geometry_motion_vae = globals()[
                    self.config.vae_type
                ].from_pretrained(
                    self.config.pretrained_vae,
                    cache_dir=self.config.cache_dir,
                    subfolder="geometry_motion_vae",
                    low_cpu_mem_usage=False,  # init temporal module
                )
            else:
                self.geometry_motion_vae = globals()[self.config.vae_type]()

        self.geometry_motion_vae.requires_grad_(False)
        self.world_map_vae_cls = self.geometry_motion_vae.__class__

        # Create EMA for the vae.
        if self.config.use_ema:
            self.ema_vae = EMAModel(
                self.geometry_motion_vae.parameters(),
                model_cls=self.world_map_vae_cls,
                model_config=self.geometry_motion_vae.config,
            )

        if self.config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 may cause issues on some GPUs. "
                        "Please upgrade to at least 0.0.17. See "
                        "https://huggingface.co/docs/diffusers/main/en/"
                        "optimization/xformers for more details."
                    )
                self.geometry_motion_vae.enable_xformers_memory_efficient_attention()
                self.vae.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.config.gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
            self.geometry_motion_vae.enable_gradient_checkpointing()
        else:
            self.vae.disable_gradient_checkpointing()
            self.geometry_motion_vae.disable_gradient_checkpointing()

    def _init_ckpt_hook(self):
        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    if self.config.use_ema:
                        self.ema_vae.save_pretrained(
                            os.path.join(output_dir, "world_map_vae_ema")
                        )

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "geometry_motion_vae"))

                        # make sure to pop weight so that corresponding model is not saved again
                        if weights:
                            weights.pop()

            def load_model_hook(models, input_dir):
                if self.config.use_ema:
                    load_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "world_map_vae_ema"),
                        self.world_map_vae_cls,
                    )
                    self.ema_vae.load_state_dict(load_model.state_dict())
                    self.ema_vae.to(self.accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    if models:
                        model = models.pop()

                    # load diffusers style into model
                    load_model = self.world_map_vae_cls.from_pretrained(
                        input_dir, subfolder="geometry_motion_vae"
                    )
                    models.register_to_config(**load_model.config)

                    models.load_state_dict(load_model.state_dict())
                    del load_model

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _init_optimizers(self):
        # Initialize optimizer with learning rate scaling
        if self.config.scale_lr:
            self.config.learning_rate = (
                self.config.learning_rate
                * self.config.gradient_accumulation_steps
                * self.config.per_gpu_batch_size
                * self.accelerator.num_processes
            )
        # Select optimizer class (8-bit Adam or regular AdamW)
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Install bitsandbytes for 8-bit Adam: "
                    "pip install bitsandbytes"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        # Configure trainable parameters based on config
        # Configure trainable parameters based on config
        self.geometry_motion_vae.requires_grad_(True)
        self.parameters_list = []
        # Select parameters to train based on train_params setting
        for name, para in self.geometry_motion_vae.named_parameters():
            if self.config.train_params == "all":
                # Train all parameters
                self.parameters_list.append(para)
                para.requires_grad = True
            elif self.config.train_params == "decoder":
                # Train decoder only
                if "decoder" in name:
                    self.parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            elif self.config.train_params == "temporal":
                # Train temporal modules only
                if "temporal" in name or 'time' in name:
                    self.parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            elif self.config.train_params == "decoder_finetune":
                # Finetune decoder
                if "decoder_finetune" in name:
                    self.parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            elif self.config.train_params == "seperate":
                # Train separate branch (encoder_2, decoder_2, etc.)
                if (
                    "encoder_2" in name or "quant_conv_2" in name or
                    "decoder_2" in name or "post_quant_conv_2" in name
                ):
                    self.parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                raise NotImplementedError

        # Create optimizer with configured parameters
        self.optimizer = optimizer_cls(
            self.parameters_list,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # Log trainable and frozen parameters
        if self.accelerator.is_main_process:
            config_save_path = (
                self.config.exp_path / "configs" /
                self.accelerator.get_tracker("wandb").run.id
            )
            config_save_path.mkdir(parents=True, exist_ok=True)
            rec_txt1 = open(
                config_save_path / "rec_param_freeze.txt", "w"
            )
            rec_txt2 = open(
                config_save_path / "rec_param_train.txt", "w"
            )
            for name, para in self.geometry_motion_vae.named_parameters():
                if para.requires_grad is False:
                    rec_txt1.write(f"{name}\n")
                else:
                    rec_txt2.write(f"{name}\n")
            rec_txt1.close()
            rec_txt2.close()

    def train(self):
        total_batch_size = (
            self.config.per_gpu_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        num_examples = (
            len(self.train_dataloader)
            * self.config.per_gpu_batch_size
            * self.accelerator.num_processes
        )
        logger.info(f"  Num examples = {num_examples}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        global_step = 0
        if self.config.resume_from_checkpoint:
            global_step = self.resume(Path(self.config.resume_from_checkpoint))
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        self.vae.train()
        self.geometry_motion_vae.train()
        while global_step < self.config.max_train_steps:
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.geometry_motion_vae):
                    loss = self.train_iter(batch)
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = {
                        k: self.accelerator.gather(
                            v.repeat(self.config.per_gpu_batch_size)
                        ).mean()
                        for k, v in loss.items()
                    }
                    self.accelerator.backward(loss["total_loss"])
                    if self.accelerator.sync_gradients and self.config.max_grad_norm:
                        self.accelerator.clip_grad_norm_(self.parameters_list, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_vae.step(self.geometry_motion_vae.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    for k, v in avg_loss.items():
                        self.accelerator.log({f"train_{k}": v.item()}, step=global_step)

                    # save checkpoints!
                    if global_step % self.config.checkpointing_steps == 0:
                        self.save_checkpoint(global_step)

                    # sample video
                    if self.accelerator.is_main_process and (
                        (global_step % self.config.validation_steps == 0)
                        or (global_step == 1 and self.config.validation_steps < 100000)
                    ):
                        logger.info(f"Running validation...")
                        self.validate(global_step)

                logs = {
                    k:v.detach().item() for k,v in avg_loss.items()
                }
                logs.update(lr=self.lr_scheduler.get_last_lr()[0])
                progress_bar.set_postfix(**logs)

                if global_step >= self.config.max_train_steps:
                    break

    @gin.configurable(module="WMapVAEKLTrainer")
    def train_iter(
        self,
        batch
    ) -> Dict[str, torch.Tensor]:
        # Single training iteration with multi-task loss computation
        # Args:
        #   batch: Batch of training data
        # Returns:
        #   Dictionary of loss values
        if self.config.empty_cache_per_iter:
            gpu_id = int(str(self.accelerator.device)[5:])
            handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            # Clear cache if memory below 1500 MB
            if meminfo.free / 1024 / 1024 < 1500:
                torch.cuda.empty_cache()

        valid_mask = (
            batch["valid_mask"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )  # valid_mask in shape of (B, T, H, W), range: [-1, 1]
        world_map = (
            batch["point_map"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )  # world_map in shape of (B, T, 3, H, W), range: [-inf, +inf]
        camera_pose = (
            batch["camera_pose"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )  # camera_pose in shape of (B, T, 4, 4), range: [-inf, +inf]
        scene_flow = (
            batch["scene_flow"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        ) if "scene_flow" in batch else None # world_map in shape of (B, T, 3, H, W), range: [-inf, +inf]
        deform_mask = (
            batch["deform_mask"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        ) if "deform_mask" in batch else None
        B, T, _, H, W = world_map.shape

        # Verify shape consistency
        expected_shape = (B, T, 3, H, W)
        assert world_map.shape == expected_shape, (
            f"world_map shape mismatch: "
            f"{world_map.shape} != {expected_shape}"
        )
        depth_map = world_map[:, :, 2]
        if scene_flow is not None:
            assert scene_flow.shape == expected_shape, (
                f"scene_flow shape mismatch: "
                f"{scene_flow.shape} != {expected_shape}"
            )
            deform_mask = deform_mask if deform_mask is not None else valid_mask
            world_map_deform = world_map + scene_flow

        check_isnan(valid_mask, 'valid_mask')
        check_isnan(world_map, 'world_map')
        check_isnan(camera_pose, 'camera_pose')
        check_isnan(depth_map, 'depth_map')

        # Encode and decode based on VAE type
        if self.config.vae_type == "AutoencoderKL":
            # Standard AutoencoderKL encoding/decoding
            if (
                self.config.train_params == "decoder" or
                self.config.train_params == "temporal"
            ):
                # Freeze encoder during decoder-only training
                with torch.inference_mode():
                    latent_dist = self.geometry_motion_vae.encode(
                        rearrange(world_map, "b t c h w -> (b t) c h w")
                    ).latent_dist
            else:
                latent_dist = self.geometry_motion_vae.encode(
                    rearrange(world_map, "b t c h w -> (b t) c h w")
                ).latent_dist

            latent = latent_dist.sample()

            rec_world_map = self.geometry_motion_vae.decode(
                latent,
            ).sample
            world_map = rearrange(world_map, "b t c h w -> b t h w c")
            rec_world_map = rearrange(rec_world_map, "(b t) c h w -> b t h w c ", t=T)

        elif self.config.vae_type == "UnifyAutoencoderKL":
            # Dual-branch VAE for world map and scene flow
            with torch.no_grad():
                latent_wmap = self.geometry_motion_vae.encode(
                    rearrange(world_map, "b t c h w -> (b t) c h w")
                ).sample()

            # Train scene flow branch
            latent_sceneflow = self.geometry_motion_vae.encode_2(
                rearrange(scene_flow, "b t c h w -> (b t) c h w")
            ).sample()

            # Decode with conditioning
            rec_deform_map = self.geometry_motion_vae.decode_2(
                latent_sceneflow,
                latent_wmap,
            )
            rec_deform_map = rearrange(
                rec_deform_map, "(b t) c h w -> b t h w c", t=T
            )
            scene_flow = rearrange(scene_flow, "b t c h w -> b t h w c") 
 
        else:
            raise NotImplementedError

        # Compute multi-task losses
        loss = 0.0
        loss_info = {}

        # World map reconstruction loss
        if self.lambda_wmap > 0.0:
            loss_world_map = weighted_mse_loss(
                rec_world_map.float(),
                world_map.float(),
                (valid_mask > 0).float()
            )
            loss += self.lambda_wmap * loss_world_map
            loss_info['l1_wmap'] = loss_world_map
            check_isnan(loss_world_map, 'loss_world_map')

        # Scene flow reconstruction loss
        if self.lambda_sceneflow > 0.0 and scene_flow is not None:
            loss_sceneflow = weighted_mse_loss(
                rec_deform_map.float(),
                scene_flow.float(),
                (deform_mask > 0).float()
            ) + 0.01 * weighted_mse_loss(
                rec_deform_map.float(),
                torch.zeros_like(scene_flow).float()
            )
            loss += self.lambda_sceneflow * loss_sceneflow
            loss_info['l1_sf'] = loss_sceneflow
            check_isnan(loss_sceneflow, 'loss_sceneflow')

        # Depth map L1 loss
        if self.lambda_l1_depth > 0.0:
            rec_depth_map = project_to_depth_map(
                rec_world_map, camera_pose
            )
            depth_map = project_to_depth_map(world_map, camera_pose)
            loss_l1_depth = weighted_l1_loss(
                rec_depth_map.unsqueeze(-1).float(),
                depth_map.unsqueeze(-1).float(),
                (valid_mask > 0).float(),
            )
            loss += self.lambda_l1_depth * loss_l1_depth
            loss_info['l1_d'] = loss_l1_depth
            check_isnan(loss_l1_depth, 'loss_l1_depth')

        # Multi-scale patch depth loss
        if self.lambda_patch_l1_depth > 0.0:
            loss_patch_l1_depth = multi_scale_weighted_depth_loss(
                rec_depth_map.float(),
                depth_map.float(),
                (valid_mask > 0).float(),
            )
            loss += self.lambda_patch_l1_depth * loss_patch_l1_depth
            loss_info['l1_pd'] = loss_patch_l1_depth
            check_isnan(loss_patch_l1_depth, 'loss_patch_l1_depth')

        # KL divergence loss
        if self.lambda_kl > 0.0:
            loss_kl = 0.5 * torch.mean(
                (
                    torch.pow(latent_dist.mean, 2) +
                    latent_dist.var - 1.0 - latent_dist.logvar
                ),
                dim=[1, 2, 3],
            ).mean()
            if torch.isinf(loss_kl).any().item():
                loss_kl = torch.zeros_like(loss_kl)
            loss += self.lambda_kl * loss_kl
            loss_info['l_kl'] = loss_kl
            check_isnan(loss_kl, 'loss_kl')

        # Surface normal consistency loss
        if self.lambda_normal > 0.0:
            loss_normal = weighted_normal_loss(
                rec_world_map.float(),
                world_map.float(),
                (valid_mask > 0).float()
            )
            loss += self.lambda_normal * loss_normal
            loss_info['l_n'] = loss_normal
            check_isnan(loss_normal, 'loss_normal')

        # Deformed normal consistency loss
        if self.lambda_deform_normal > 0.0 and scene_flow is not None:
            loss_deform_normal = weighted_normal_loss(
                (world_map + rec_deform_map).float(),
                world_map_deform.float(),
                (deform_mask > 0).float()
            )
            loss += self.lambda_deform_normal * loss_deform_normal
            loss_info['l_dn'] = loss_deform_normal
            check_isnan(loss_deform_normal, 'loss_deform_normal')

        loss_info['total_loss'] = loss
        return loss_info

    def validate(
        self,
        global_step,
    ):
        # Validation step (placeholder)
        # Can be implemented for generating validation samples
        torch.cuda.empty_cache()
        return