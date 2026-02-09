"""Deterministic UNet Trainer - single-step inference without diffusion."""

from typing import Dict
import os
import pynvml
import torch
import torch.nn as nn
import gin

from accelerate.logging import get_logger

from .unet_base import UNetFullDeformBaseTrainer

logger = get_logger(__name__, log_level="INFO")


@gin.configurable()
class UNetFullDeformDetermTrainer(UNetFullDeformBaseTrainer):
    """
    Deterministic UNet trainer for single-step motion generation.
    
    Inherits shared training logic from base class and implements
    deterministic inference without iterative denoising.
    """

    def _get_unet_subfolder(self, pretrained_path):
        """Get UNet subfolder for deterministic model."""
        subfolder_path = os.path.join(pretrained_path, "unet_determ")
        return "unet_determ" if os.path.exists(subfolder_path) else None

    def _check_unet_channels(self):
        """Check if UNet has correct 4 input channels for deterministic model."""
        return self.unet.config["in_channels"] == 4

    def _replace_unet_conv_in(self):
        """
        Replace first conv layer to accept 4 RGB conditioning channels.
        Extracts RGB condition channels from pretrained 8-channel weights.
        """
        _weight = self.unet.conv_in.weight.clone()  # [320, 8, 3, 3]
        _bias = self.unet.conv_in.bias.clone()  # [320]
        # Use RGB condition channels (channels 4-8 from original)
        _weight = _weight[:, 4:8, :, :]
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
        logger.info("Unet conv_in layer replaced for deterministic mode (4 channels)")
        # Update config
        self.unet.config["in_channels"] = _weight.shape[1]
        logger.info("Unet config updated")


    def train_iter(
        self,
        batch,
        noisy_conditions: bool = True,
        chunk_size: int = 25,
        low_quality_threshold: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """
        Single deterministic training iteration.
        
        Uses fixed timestep and single-step prediction without diffusion.
        
        Args:
            batch: Batch of training data
            noisy_conditions: Enable noise augmentation
            chunk_size: VAE encoding chunk size
            low_quality_threshold: Frame quality threshold
            
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
        
        # Fixed timestep for deterministic mode
        timesteps = torch.Tensor([1.6378] * B).to(self.accelerator.device)

        # Prepare time embeddings
        added_time_ids, _, _ = self._prepare_time_embeddings(noise_aug_strength, B)
        self._validate_time_embedding_dims(added_time_ids)
        added_time_ids = torch.cat(added_time_ids, dim=1).to(target_latents)

        # Forward pass
        model_pred = self.unet(
            conditional_latents,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
        ).sample

        # Denoise with fixed c_out factor
        denoised_latents = model_pred * -1

        # Compute losses
        loss = {}
        
        # Latent space losses
        if self.lambda_latent > 0.0:
            loss["latent_wmap"] = self.lambda_latent * torch.mean(
                (denoised_latents[:, :, :4].float() - target_latents[:, :, :4].float()) ** 2
            )
            if scene_flow is not None:
                loss["latent_sceneflow"] = self.lambda_latent * torch.mean(
                    (denoised_latents[:, :-1, 4:].float() - target_latents[:, :-1, 4:].float()) ** 2
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

    @gin.configurable(module="UNetFullDeformDetermTrainer")
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
        Can be implemented to generate validation samples.
        """
        torch.cuda.empty_cache()
        return

