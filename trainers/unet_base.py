"""Base UNet Trainer with shared training logic for deterministic and diffusion models."""

from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod
import os
import random
from pathlib import Path
import pynvml
import torch
import torch.nn as nn
import accelerate
import xformers

from accelerate.logging import get_logger
from packaging import version
from tqdm.auto import tqdm
import gin

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing
)

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

from motioncrafter import *
from .base import BaseTrainer
from utils.losses import weighted_mse_loss, chamfer_distance
import torch.nn.functional as F

# Check minimum diffusers version
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def rand_log_normal(
    shape, loc=0.0, scale=1.0, device='cpu', dtype=torch.float32
):
    """
    Draw samples from lognormal distribution.
    Used for noise augmentation in conditioning and diffusion process.
    
    Args:
        shape: Output tensor shape
        loc: Mean of the underlying normal distribution
        scale: Standard deviation of the underlying normal distribution
        device: Device to create tensor on
        dtype: Data type of output tensor
        
    Returns:
        Samples from log-normal distribution
    """
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def filter_low_quality_frame(valid_mask, threshold=0.02):
    """
    Filter frames with low valid pixel counts.
    
    Args:
        valid_mask: Frame validity mask (B, T, H, W)
        threshold: Minimum valid pixel ratio (default 0.02 = 2%)
        
    Returns:
        Boolean tensor indicating high-quality frames
    """
    B, T, H, W = valid_mask.shape
    assert B == 1
    selected_indices = []
    for i in range(T):
        if (valid_mask[0, i] > 0).sum() > threshold * H * W:
            selected_indices.append(True)
        else:
            selected_indices.append(False)
    return torch.tensor(
        selected_indices, dtype=torch.bool, device=valid_mask.device
    )


class UNetFullDeformBaseTrainer(BaseTrainer, ABC):
    """
    Base trainer for UNet motion generation models.
    
    Provides shared functionality for both deterministic and diffusion-based
    training including model initialization, encoding/decoding, and training loop.
    """
    
    def __init__(
        self,
        args: dataclass,
        lambda_latent: float = 1.0,
        lambda_wmap: float = 1.0,
        lambda_deform_mask: float = 0.1,
        lambda_valid_mask: float = 0.1,
        lambda_sceneflow: float = 1.0,
        lambda_chamfer: float = 1.0,
    ):
        """
        Initialize base trainer with loss weights.
        
        Args:
            args: Training configuration
            lambda_*: Loss weights for different components
        """
        super().__init__(args)

        self.lambda_latent = lambda_latent
        self.lambda_wmap = lambda_wmap
        self.lambda_deform_mask = lambda_deform_mask
        self.lambda_valid_mask = lambda_valid_mask
        self.lambda_sceneflow = lambda_sceneflow
        self.lambda_chamfer = lambda_chamfer

        self._init_models()
        self._init_ckpt_hook()
        self._init_optimizers()

        pynvml.nvmlInit()
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
        )

        # Prepare everything with accelerator
        self.unet, self.optimizer, self.lr_scheduler, self.train_dataloader = (
            self.accelerator.prepare(
                self.unet, self.optimizer, self.lr_scheduler, self.train_dataloader
            )
        )
        if args.use_ema:
            self.ema_unet.to(self.accelerator.device)

    @abstractmethod
    def _replace_unet_conv_in(self):
        """Replace UNet input conv layer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _replace_unet_conv_out(self):
        """Replace UNet output conv layer. Must be implemented by subclasses."""
        pass

    def _init_models(self):
        """Initialize all models: feature extractor, encoders, VAEs, and UNet."""
        assert self.config.pretrained_model_name_or_path is not None
        
        # Initialize CLIP feature extractor
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.config.pretrained_model_name_or_path,
            cache_dir=self.config.cache_dir,
            subfolder="feature_extractor",
            revision=self.config.revision,
        )
        
        # Initialize CLIP image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            cache_dir=self.config.cache_dir,
            subfolder="image_encoder",
            revision=self.config.revision,
            variant="fp16",  # fix for inference, only point map enc-dec requires fp32
        )
        
        # Initialize RGB VAE
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.config.pretrained_model_name_or_path,
            cache_dir=self.config.cache_dir,
            subfolder="vae",
            revision=self.config.revision,
            variant="fp16" if self.config.mixed_precision == 'fp16' else None,
        )

        # Freeze image encoder and enable gradient checkpointing for VAE
        self.image_encoder.requires_grad_(False)
        self.image_encoder.to(self.accelerator.device, dtype=torch.float16)
        self.vae.requires_grad_(False)
        self.vae.decoder.train()  # enable grad checkpoint
        self.vae._set_gradient_checkpointing(self.vae.decoder, True)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # Initialize point map VAE
        assert self.config.pretrained_vae is not None
        self.geometry_motion_vae = globals()[self.config.vae_type].from_pretrained(
            self.config.pretrained_vae,
            cache_dir=self.config.cache_dir,
            subfolder="geometry_motion_vae",
            low_cpu_mem_usage=True,
        )
        self.geometry_motion_vae.requires_grad_(False)
        self.geometry_motion_vae.decoder.train()  # enable grad checkpoint
        self.geometry_motion_vae._set_gradient_checkpointing(self.geometry_motion_vae.decoder, True)
        self.geometry_motion_vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # Initialize UNet (subclass determines subfolder)
        pretrained_path = (
            self.config.pretrained_model_name_or_path
            if self.config.pretrained_unet is None
            else self.config.pretrained_unet
        )
        subfolder = self._get_unet_subfolder(pretrained_path)
        
        self.unet = globals()[self.config.unet_type].from_pretrained(
            pretrained_path,
            cache_dir=self.config.cache_dir,
            subfolder=subfolder,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if self.config.mixed_precision == 'fp16' else torch.float32,
            variant="fp16" if self.config.mixed_precision == 'fp16' else None,
        )
        
        # Replace conv layers if needed
        if not self._check_unet_channels():
            self._replace_unet_conv_in()
            self._replace_unet_conv_out()

        self.unet.requires_grad_(False)
        self.unet_cls = self.unet.__class__
        
        # Create EMA for the unet
        if self.config.use_ema:
            self.ema_unet = EMAModel(
                self.unet.parameters(),
                model_cls=self.unet_cls,
                model_config=self.unet.config,
            )

        # Enable xformers if available
        if self.config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 may cause training issues on some GPUs. "
                        "Please upgrade to at least 0.0.17."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        # Enable/disable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()

    @abstractmethod
    def _get_unet_subfolder(self, pretrained_path):
        """Get UNet subfolder name. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _check_unet_channels(self):
        """Check if UNet has correct input channels. Must be implemented by subclasses."""
        pass

    def _init_ckpt_hook(self):
        """Initialize checkpoint save/load hooks."""
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    if self.config.use_ema:
                        self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

            def load_model_hook(models, input_dir):
                if self.config.use_ema:
                    load_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "unet_ema"),
                        self.unet_cls
                    )
                    self.ema_unet.load_state_dict(load_model.state_dict())
                    self.ema_unet.to(self.accelerator.device)
                    del load_model

                for i in range(len(models)):
                    models[i].__class__.from_pretrained(
                        input_dir, subfolder="unet"
                    )

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _init_optimizers(self):
        """Initialize optimizer and prepare trainable parameters."""
        if self.config.scale_lr:
            self.config.learning_rate = (
                self.config.learning_rate
                * self.config.gradient_accumulation_steps
                * self.config.per_gpu_batch_size
                * self.accelerator.num_processes
            )
        
        # Select optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. "
                    "You can do so by running `pip install bitsandbytes`"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        # Prepare trainable parameters
        self.unet.requires_grad_(True)
        self.parameters_list = []
        
        for name, para in self.unet.named_parameters():
            if self.config.train_params == 'all':
                self.parameters_list.append(para)
                para.requires_grad = True
            elif self.config.train_params == "temporal":
                if "temporal" in name:
                    self.parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                raise NotImplementedError

        self.optimizer = optimizer_cls(
            self.parameters_list,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # Log trainable parameters
        if self.accelerator.is_main_process:
            config_save_path = (
                self.config.exp_path / "configs" / 
                self.accelerator.get_tracker("wandb").run.id
            )
            config_save_path.mkdir(parents=True, exist_ok=True)
            rec_txt1 = open(config_save_path / "rec_param_freeze.txt", "w")
            rec_txt2 = open(config_save_path / "rec_param_train.txt", "w")
            for name, para in self.unet.named_parameters():
                if para.requires_grad is False:
                    rec_txt1.write(f"{name}\n")
                else:
                    rec_txt2.write(f"{name}\n")
            rec_txt1.close()
            rec_txt2.close()

    def train(self):
        """Main training loop."""
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
        
        progress_bar = tqdm(
            range(global_step, self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        self.unet.train()
        while global_step < self.config.max_train_steps:
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    avg_loss = self.train_iter(batch)
                    if avg_loss is None:
                        continue
                    loss = avg_loss["total_loss"]
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.parameters_list, self.config.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % self.config.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = (
                                self.config.exp_path / f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    if global_step % self.config.validation_steps == 0:
                        if self.accelerator.is_main_process:
                            self.validate(global_step)

                    self.accelerator.log(avg_loss, step=global_step)

                logs = {k: v.detach().item() for k, v in avg_loss.items()}
                logs.update(lr=self.lr_scheduler.get_last_lr()[0])
                progress_bar.set_postfix(**logs)

                if global_step >= self.config.max_train_steps:
                    break

    @torch.no_grad()
    def encode_vae_video(self, frame, chunk_size=8, noisy_conditions=True):
        """
        Encode video frames to VAE latents with optional noise augmentation.
        
        Args:
            frame: Input video frames (B, T, 3, H, W)
            chunk_size: Processing chunk size for memory efficiency
            noisy_conditions: Whether to add noise to frames
            
        Returns:
            conditional_latents: Encoded video latents (B, T, c, h, w)
            cond_sigmas: Noise augmentation strengths (B, 1, 1, 1, 1)
        """
        B, T, _, H, W = frame.shape
        if noisy_conditions:
            cond_sigmas = rand_log_normal(
                shape=[B,], loc=-3.0, scale=0.5,
            ).to(frame)
            cond_sigmas = cond_sigmas[:, None, None, None, None]
            conditionals = torch.randn_like(frame) * cond_sigmas + frame
        else:
            raise NotImplementedError

        latents = []
        cond_ = rearrange(conditionals, "b t c h w -> (b t) c h w")
        for i in range(0, B * T, chunk_size):
            latents.append(
                self.vae.encode(cond_[i : i + chunk_size]).latent_dist.sample()
            )
        conditional_latents = torch.cat(latents, dim=0)
        conditional_latents = rearrange(
            conditional_latents, "(b t) c h w -> b t c h w", t=T
        )
        return conditional_latents, cond_sigmas

    @torch.no_grad()
    def text_embed_video(self, frame, chunk_size=8):
        """
        Extract CLIP embeddings from video frames.
        
        Args:
            frame: Input video frames (B, T, 3, H, W)
            chunk_size: Processing chunk size
            
        Returns:
            encoder_hidden_states: CLIP embeddings (B, T, 1024)
        """
        B, T, _, H, W = frame.shape
        conditionals_224 = rearrange(
            frame.to(self.image_encoder.dtype), "b t c h w -> (b t) c h w"
        )
        encoder_hidden_states = []
        for i in range(0, conditionals_224.shape[0], chunk_size):
            video = conditionals_224[i : i + chunk_size]
            video_224 = _resize_with_antialiasing(video.float(), (224, 224))
            video_224 = (video_224 + 1.0) / 2.0
            video_224 = self.feature_extractor(
                images=video_224,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings = self.image_encoder(video_224).image_embeds
            encoder_hidden_states.append(embeddings)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)
        encoder_hidden_states = rearrange(
            encoder_hidden_states, "(b t) c -> b t c", t=T
        )
        return encoder_hidden_states

    @torch.no_grad()
    def encode_point_map(
        self, valid_mask, point_map, scene_flow=None, chunk_size=8
    ):
        """
        Encode point map and scene flow to VAE latents.
        
        Args:
            valid_mask: Validity mask (B, T, H, W)
            point_map: 3D point coordinates (B, T, 3, H, W)
            scene_flow: 3D motion vectors (B, T, 3, H, W) optional
            chunk_size: Processing chunk size
            
        Returns:
            latents: Encoded latents (B, T, c, h, w)
        """
        B, T, _, H, W = point_map.shape
        latents = []

        pmap_ = rearrange(point_map, "b t c h w -> (b t) c h w")
        vmask_ = rearrange(valid_mask, "b t h w -> (b t) h w")
        if scene_flow is not None:
            sflow_ = rearrange(scene_flow, "b t c h w -> (b t) c h w")
        else:
            sflow_ = torch.zeros_like(pmap_)

        for i in range(0, B * T, chunk_size):
            if self.config.vae_type == "AutoencoderKL":
                latent = self.geometry_motion_vae.encode(
                    pmap_[i : i + chunk_size]
                ).latent_dist.mode()
                latent_deform = self.geometry_motion_vae.encode(
                    sflow_[i : i + chunk_size]
                ).latent_dist.mode()
                latent = torch.cat([latent, latent_deform], dim=1)
            elif self.config.vae_type == "SeperateAutoencoderKL":
                latent = self.geometry_motion_vae.encode(pmap_[i : i + chunk_size])
                latent_deform = self.geometry_motion_vae.encode_2(sflow_[i : i + chunk_size])
                latent = torch.cat([latent, latent_deform], dim=1)
            elif self.config.vae_type == "UnifyAutoencoderKL":
                latent = self.geometry_motion_vae.encode(
                    pmap_[i : i + chunk_size],
                    sflow_[i : i + chunk_size],
                    vmask_[i : i + chunk_size],
                )
            assert isinstance(latent, torch.Tensor)
            latents.append(latent)
            
        latents = torch.cat(latents, dim=0)
        latents = rearrange(latents, "(b t) c h w -> b t c h w", t=T)
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_point_map(self, latents, chunk_size=8):
        """
        Decode VAE latents back to point map and scene flow.
        
        Args:
            latents: Encoded latents (B, T, c, h, w)
            chunk_size: Processing chunk size
            
        Returns:
            rec_point_maps: Reconstructed point maps (B, T, 3, H, W)
            rec_valid_masks: Reconstructed validity masks (B, T, H, W)
            rec_deform_maps: Reconstructed deformation/scene flow
        """
        B, T, _, H, W = latents.shape
        latents = rearrange(latents, "b t c h w -> (b t) c h w")
        rec_point_maps = []
        rec_valid_masks = []
        rec_deform_maps = []
        
        for i in range(0, B * T, chunk_size):
            lat = latents[i : i + chunk_size]
            lat_1 = lat[:, :4, :, :]
            lat_2 = lat[:, 4:8, :, :]

            if self.config.vae_type == "AutoencoderKL":
                rec_pointmap = self.geometry_motion_vae.decode(lat_1).sample
                rec_vmask = torch.ones_like(rec_pointmap[:, :1, :, :])
            elif self.config.vae_type == "SeperateAutoencoderKL":
                rec_pointmap = self.geometry_motion_vae.decode(lat_1)
                rec_vmask = torch.ones_like(rec_pointmap[:, :1, :, :])
                rec_deformmap = self.geometry_motion_vae.decode_2(lat_2)
            elif self.config.vae_type == "UnifyAutoencoderKL":
                rec_pointmap = self.geometry_motion_vae.decode(lat_1)
                rec_vmask = torch.ones_like(rec_pointmap[:, :1, :, :])
                rec_deformmap = self.geometry_motion_vae.decode_2(lat_2, lat_1)

            rec_point_maps.append(rec_pointmap)
            rec_valid_masks.append(rec_vmask)
            rec_deform_maps.append(rec_deformmap)

        rec_point_maps = torch.cat(rec_point_maps, dim=0)
        rec_valid_masks = torch.cat(rec_valid_masks, dim=0)
        rec_deform_maps = torch.cat(rec_deform_maps, dim=0)

        rec_point_maps = rearrange(rec_point_maps, "(b t) c h w -> b t c h w", t=T)
        rec_valid_masks = rearrange(rec_valid_masks, "(b t) 1 h w -> b t h w", t=T) > 0
        rec_deform_maps = rearrange(rec_deform_maps, "(b t) c h w -> b t c h w", t=T)

        return rec_point_maps, rec_valid_masks, rec_deform_maps

    @abstractmethod
    def train_iter(
        self,
        batch,
        noisy_conditions: bool = True,
        chunk_size: int = 25,
        low_quality_threshold: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training iteration. Must be implemented by subclasses.
        
        Args:
            batch: Batch of training data
            noisy_conditions: Enable noise augmentation
            chunk_size: VAE encoding chunk size
            low_quality_threshold: Frame quality threshold
            
        Returns:
            Dictionary of loss values
        """
        pass

    def _replace_unet_conv_out(self):
        """
        Replace output conv layer to output 8 channels.
        Outputs: point map (4ch) + scene flow (4ch)
        Shared implementation for both deterministic and diffusion trainers.
        """
        _weight = self.unet.conv_out.weight.clone()
        _bias = self.unet.conv_out.bias.clone()
        _weight = torch.cat(
            [_weight[0:4, :, :, :], _weight[0:4, :, :, :]],  # duplicate
            dim=0
        )
        _bias = torch.cat([_bias[0:4], _bias[0:4]], dim=0)
        _new_conv_out = nn.Conv2d(
            320, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_out.weight = nn.Parameter(_weight)
        _new_conv_out.bias = nn.Parameter(_bias)
        self.unet.conv_out = _new_conv_out
        logger.info("Unet conv_out layer replaced (8 channels)")
        self.unet.config["out_channels"] = 8

    def _compute_reconstruction_losses(
        self,
        denoised_latents,
        target_latents,
        valid_mask,
        point_map,
        scene_flow,
    ):
        """
        Compute reconstruction losses in output space.
        Shared implementation for loss computation across trainers.
        
        Args:
            denoised_latents: Denoised VAE latents (B, T, 8, h, w)
            target_latents: Ground truth latents (B, T, 8, h, w)
            valid_mask: Validity mask (B, T, H, W)
            point_map: Point coordinates (B, T, 3, H, W)
            scene_flow: Motion vectors (B, T, 3, H, W) or None
            
        Returns:
            Dictionary of computed loss tensors
        """
        loss = {}
        
        # Decode latents to output space
        denoised_latents = 1 / self.vae.config.scaling_factor * denoised_latents
        rec_point_map, rec_valid_mask, rec_deform_map = self.decode_point_map(
            denoised_latents, chunk_size=8
        )

        # Rearrange to HWC format
        rec_point_map = rearrange(rec_point_map, "b t c h w -> b t h w c")
        point_map = rearrange(point_map, "b t c h w -> b t h w c")
        rec_deform_map = rearrange(rec_deform_map, "b t c h w -> b t h w c")
        if scene_flow is not None:
            scene_flow = rearrange(scene_flow, "b t c h w -> b t h w c")

        # Point map reconstruction loss
        if self.lambda_wmap > 0.0:
            loss["world_map"] = self.lambda_wmap * weighted_mse_loss(
                rec_point_map.float(),
                point_map.float(),
                (valid_mask > 0).float()
            )

        # Valid mask reconstruction loss
        if self.lambda_valid_mask > 0.0:
            loss["rec_valid_mask"] = self.lambda_valid_mask * F.mse_loss(
                rec_valid_mask.float(),
                valid_mask.float(),
                reduction='mean'
            )

        # Scene flow reconstruction loss
        if self.lambda_sceneflow > 0.0 and scene_flow is not None:
            loss["sceneflow"] = self.lambda_sceneflow * weighted_mse_loss(
                rec_deform_map[:, :-1].float(),
                scene_flow[:, :-1].float(),
                (valid_mask[:, :-1] > 0).float()
            )

        # Chamfer distance between consecutive frames
        if self.lambda_chamfer > 0.0:
            loss["chamfer"] = self.lambda_chamfer * chamfer_distance(
                rec_point_map[:, 1:].float(),
                (rec_point_map[:, :-1] + rec_deform_map[:, :-1]).float(),
                valid_mask[:, 1:].float(),
                valid_mask[:, :-1].float(),
            )

        return loss

    def _load_batch_data(self, batch):
        """
        Load and transfer batch data to device.
        
        Returns:
            Tuple of (frame, valid_mask, point_map, scene_flow)
        """
        frame = (
            batch["frame"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )
        valid_mask = (
            batch["valid_mask"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )
        point_map = (
            batch["point_map"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        )
        scene_flow = (
            batch["scene_flow"]
            .to(self.weight_dtype)
            .to(self.accelerator.device, non_blocking=True)
        ) if 'scene_flow' in batch else None
        
        return frame, valid_mask, point_map, scene_flow

    def _encode_batch_data(self, frame, valid_mask, point_map, scene_flow, chunk_size):
        """
        Encode batch data (CLIP embeddings, VAE latents, point maps).
        
        Returns:
            Tuple of (target_latents, encoder_hidden_states, conditional_latents,
                     noise_aug_strength, B, T, H, W)
        """
        # Encode ground truth
        target_latents = self.encode_point_map(
            valid_mask, point_map, scene_flow=scene_flow, chunk_size=chunk_size
        )

        # Extract CLIP embeddings
        encoder_hidden_states = self.text_embed_video(frame, chunk_size=chunk_size)

        # Encode RGB frames with noise augmentation
        conditional_latents, noise_aug_strength = self.encode_vae_video(
            frame, chunk_size=chunk_size, noisy_conditions=True
        )

        B, T, _, H, W = target_latents.shape
        return (
            target_latents, encoder_hidden_states, conditional_latents,
            noise_aug_strength, B, T, H, W
        )

    def _prepare_time_embeddings(self, noise_aug_strength, B):
        """
        Prepare time embeddings (fps, motion_bucket_id, noise_aug_strength).
        
        Returns:
            Tuple of (added_time_ids, fps, motion_bucket_id)
        """
        fps = 7
        motion_bucket_id = 127
        added_time_ids = [
            fps * torch.ones_like(noise_aug_strength.reshape(B, 1)),
            motion_bucket_id * torch.ones_like(noise_aug_strength.reshape(B, 1)),
            noise_aug_strength.reshape(B, 1)
        ]
        return added_time_ids, fps, motion_bucket_id

    def _validate_time_embedding_dims(self, added_time_ids):
        """Validate time embedding dimensions match UNet configuration."""
        passed_add_embed_dim = (
            self.unet.module.config.addition_time_embed_dim * len(added_time_ids)
        )
        expected_add_embed_dim = (
            self.unet.module.add_embedding.linear_1.in_features
        )
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects time embedding of length {expected_add_embed_dim}, "
                f"got {passed_add_embed_dim}."
            )

    @abstractmethod
    def validate(self, global_step, **kwargs):
        """Validation step. Must be implemented by subclasses."""
        pass
