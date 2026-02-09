"""Diffusion UNet Trainer - iterative denoising for motion generation."""

from typing import Dict
import os
import pynvml
import torch
import torch.nn as nn
import gin

from accelerate.logging import get_logger

from .unet_base import UNetFullDeformBaseTrainer, rand_log_normal

logger = get_logger(__name__, log_level="INFO")


@gin.configurable()
class UNetFullDeformDiffuseTrainer(UNetFullDeformBaseTrainer):
    """
    Diffusion UNet trainer for iterative denoising motion generation.
    
    Inherits shared training logic from base class and implements
    diffusion-based training with noise prediction.
    """

    def __init__(
        self,
        *args,
        num_ddim_timesteps: int = 25,
        **kwargs
    ):
        """
        Initialize diffusion trainer.
        
        Args:
            num_ddim_timesteps: Number of DDIM sampling steps for inference
        """
        super().__init__(*args, **kwargs)
        self.num_ddim_timesteps = num_ddim_timesteps

    def _get_unet_subfolder(self, pretrained_path):
        """Get UNet subfolder for diffusion model."""
        subfolder_path = os.path.join(pretrained_path, "unet_diff")
        return "unet_diff" if os.path.exists(subfolder_path) else None

    def _check_unet_channels(self):
        """Check if UNet has correct 12 input channels for diffusion model."""
        return self.unet.config["in_channels"] == 12

    def _replace_unet_conv_in(self):
        """
        Replace first conv layer to accept 12 channels:
        - 4ch: noisy point map + scene flow latents
        - 4ch: RGB condition latents
        - 4ch: prior (point map from previous frame)
        
        Initializes noisy channels from pretrained weights, others from zero.
        """
        _weight = self.unet.conv_in.weight.clone()  # [320, 8, 3, 3]
        _bias = self.unet.conv_in.bias.clone()  # [320]
        # Initialize with zeros for new channels
        _weight = torch.cat(
            [
                _weight[:, 0:4, :, :],  # Noisy latent channels
                _weight[:, 4:8, :, :],  # RGB condition
                torch.zeros_like(_weight[:, 0:4, :, :]),  # Prior (zero-initialized)
            ],
            dim=1
        )
        _n_conv_in_out_channel = self.unet.conv_in.out_channels
        _new_conv_in = nn.Conv2d(
            _weight.shape[1],
            _n_conv_in_out_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        _new_conv_in.weight = nn.Parameter(_weight)
        _new_conv_in.bias = nn.Parameter(_bias)
        self.unet.conv_in = _new_conv_in
        logger.info("Unet conv_in layer replaced for diffusion mode (12 channels)")
        # Update config
        self.unet.config["in_channels"] = _weight.shape[1]
        logger.info("Unet config updated")

    def _replace_unet_conv_out(self):
        """
        Replace output conv layer to output 8 channels.
        Uses inherited implementation from base class.
        """
        super()._replace_unet_conv_out()

    def train_iter(
        self,
        batch,
        noisy_conditions: bool = True,
        chunk_size: int = 25,
        low_quality_threshold: float = 0.02,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        p_mean: float = -3.0,
        p_std: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Single diffusion training iteration.
        
        Uses EDM-style noise schedules and classifier-free guidance.
        
        Args:
            batch: Batch of training data
            noisy_conditions: Enable noise augmentation
            chunk_size: VAE encoding chunk size
            low_quality_threshold: Frame quality threshold
            min_guidance_scale: Minimum guidance scale for CFG
            max_guidance_scale: Maximum guidance scale for CFG
            p_mean: Mean for log-normal noise sampling
            p_std: Std for log-normal noise sampling
            
        Returns:
            Dictionary of loss values
        """
        # Memory management
        if self.config.empty_cache_per_iter:
            gpu_id = int(str(self.accelerator.device)[5:])
            handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            if meminfo.free / 1024 / 1024 < 1500:
                torch.cuda.empty_cache()

        # Load and encode batch data
        frame, valid_mask, point_map, scene_flow = self._load_batch_data(batch)
        (target_latents, encoder_hidden_states, conditional_latents,
         noise_aug_strength, B, T, H, W) = self._encode_batch_data(
            frame, valid_mask, point_map, scene_flow, chunk_size
        )
        
        # Sample random timesteps from log-normal distribution (EDM style)
        sigmas = rand_log_normal(
            shape=[B,], loc=p_mean, scale=p_std
        ).to(self.accelerator.device)
        timesteps = sigmas * 1000

        # Prepare time embeddings
        added_time_ids, _, _ = self._prepare_time_embeddings(noise_aug_strength, B)
        self._validate_time_embedding_dims(added_time_ids)
        added_time_ids = torch.cat(added_time_ids, dim=1).to(target_latents)

        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        sigmas_reshaped = sigmas.view(-1, 1, 1, 1, 1)
        noisy_latents = target_latents + noise * sigmas_reshaped

        # Prepare prior: use first frame's point map for all frames
        prior_latents = torch.zeros_like(target_latents)
        prior_latents[:, 1:, :4] = target_latents[:, :-1, :4]

        # Concatenate: [noisy, condition, prior]
        input_latents = torch.cat(
            [noisy_latents, conditional_latents, prior_latents], dim=2
        )

        # Classifier-free guidance: randomly drop conditioning
        guidance_scale = torch.rand(B, device=self.accelerator.device) * (
            max_guidance_scale - min_guidance_scale
        ) + min_guidance_scale
        guidance_scale = guidance_scale.view(-1, 1, 1, 1, 1)
        
        # 10% unconditioned for CFG training
        uncond_mask = (torch.rand(B, device=self.accelerator.device) < 0.1).float()
        uncond_mask = uncond_mask.view(-1, 1, 1, 1, 1)
        input_latents[:, :, 4:8] = input_latents[:, :, 4:8] * (1 - uncond_mask)

        # Forward pass - predict noise
        model_pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
        ).sample

        # EDM loss weighting
        c_skip = 1 / (sigmas_reshaped ** 2 + 1)
        c_out = -sigmas_reshaped / (sigmas_reshaped ** 2 + 1) ** 0.5
        denoised_latents = c_out * model_pred + c_skip * noisy_latents

        # Compute losses
        loss = {}
        
        # Latent space losses (weighted by guidance scale)
        if self.lambda_latent > 0.0:
            loss["latent_wmap"] = self.lambda_latent * torch.mean(
                guidance_scale * (
                    (denoised_latents[:, :, :4].float() - target_latents[:, :, :4].float()) ** 2
                )
            )
            if scene_flow is not None:
                loss["latent_sceneflow"] = self.lambda_latent * torch.mean(
                    guidance_scale * (
                        (denoised_latents[:, :-1, 4:].float() - target_latents[:, :-1, 4:].float()) ** 2
                    )
                )
            else:
                loss["latent_sceneflow"] = torch.tensor(0.0, device=denoised_latents.device)

        # Reconstruction losses in output space
        rec_losses = self._compute_reconstruction_losses(
            denoised_latents, target_latents, valid_mask, point_map, scene_flow
        )
        loss.update(rec_losses)

        loss["total_loss"] = sum(loss.values())
        return loss

    @gin.configurable(module="UNetFullDeformDiffuseTrainer")
    def validate(
        self,
        global_step,
        width: int = 576,
        height: int = 320,
        test_case: str = "",
        vid_length: int = 14,
        decode_chunk_size: int = 8
    ):
        """
        Validation step (currently a placeholder).
        Can be implemented to generate validation samples using DDIM sampling.
        """
        torch.cuda.empty_cache()
        return
