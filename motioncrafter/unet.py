"""UNet Spatio-Temporal Condition Model for Video-to-Video Generation.

This module extends the diffusers UNetSpatioTemporalConditionModel to support:
- Gradient checkpointing for memory-efficient training
- Frame-wise processing with temporal attention
- Custom forward pass for motion generation tasks

The model architecture is based on Stable Video Diffusion's UNet with modifications
for processing video frames with spatio-temporal conditioning.
"""

from typing import Union, Tuple

import torch
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionOutput
)
from diffusers.utils import is_torch_version


class UNetSpatioTemporalConditionModelVid2vid(
    UNetSpatioTemporalConditionModel
):
    """Extended UNet for spatio-temporal video generation with gradient checkpointing.
    
    This model extends the base UNetSpatioTemporalConditionModel to add:
    - Gradient checkpointing capability to reduce memory during training
    - Compatibility with MotionCrafter's video processing pipeline
    
    The model processes video frames with temporal attention mechanisms and
    supports both training and inference modes with configurable memory usage.
    
    Attributes:
        gradient_checkpointing (bool): Whether gradient checkpointing is enabled.
    """
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training.
        
        When enabled, intermediate activations are not stored during the forward pass.
        Instead, they are recomputed during the backward pass, trading computation
        for memory. This is especially useful for training on high-resolution videos
        or with large batch sizes.
        """
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for faster inference.
        
        Disabling gradient checkpointing reduces computation time during inference
        by storing all intermediate activations instead of recomputing them.
        This is the default mode for inference.
        """
        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        """Forward pass for spatio-temporal UNet with video frame processing.
        
        Processes video frames through a spatio-temporal UNet architecture with
        temporal attention and cross-attention mechanisms. The model handles both
        spatial and temporal dimensions for video-to-video generation tasks.
        
        Args:
            sample: Input video frames of shape [batch, frames, channels, height, width].
                    For MotionCrafter, channels can be 4 (deterministic) or 12 (diffusion).
            timestep: Diffusion timestep, either as tensor, float, or int. Used to
                     condition the denoising process.
            encoder_hidden_states: Image embeddings for cross-attention of shape
                                  [batch, frames, embedding_dim]. These provide
                                  semantic guidance from the input frames.
            added_time_ids: Additional time embeddings for conditioning, typically
                           containing augmentation parameters like noise strength.
            return_dict: Whether to return output as UNetSpatioTemporalConditionOutput
                        object (True) or as tuple (False).
        
        Returns:
            If return_dict is True:
                UNetSpatioTemporalConditionOutput with 'sample' field containing
                denoised video frames of shape [batch, frames, channels, height, width].
            If return_dict is False:
                Tuple containing the denoised frames.
                
        Note:
            - The input sample is reshaped from [B, T, C, H, W] to [B*T, C, H, W] for
              processing, then reshaped back after UNet processing.
            - Timestep embeddings are expanded and repeated for each frame.
            - Encoder hidden states are flattened and unsqueezed for cross-attention.
        """
        # 1. Process timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # Convert scalar timestep to tensor with appropriate dtype
            # Handle MPS device compatibility issues
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            # Handle scalar tensor case
            timesteps = timesteps[None].to(sample.device)

        # Expand timestep to match batch size (one timestep per batch item)
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        # Project timesteps to embedding space using sinusoidal position encoding
        t_emb = self.time_proj(timesteps)
        # Ensure timestep embeddings match model dtype
        t_emb = t_emb.to(dtype=self.conv_in.weight.dtype)

        # Process through time embedding MLP
        emb = self.time_embedding(t_emb)  # [batch_size, embedding_channels]

        # Add augmentation embeddings (e.g., noise strength, motion scale)
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # 2. Reshape for frame-wise processing
        # Flatten batch and time dimensions: [B, T, C, H, W] -> [B*T, C, H, W]
        sample = sample.flatten(0, 1)
        # Repeat embeddings for each frame: [B, D] -> [B*T, D]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # Prepare encoder hidden states for cross-attention
        # [B, T, D] -> [B*T, D] -> [B*T, 1, D]
        encoder_hidden_states = (
            encoder_hidden_states.flatten(0, 1).unsqueeze(1)
        )

        # 3. Pre-process: initial convolution
        sample = sample.to(dtype=self.conv_in.weight.dtype)
        assert sample.dtype == self.conv_in.weight.dtype, (
            f"Data type mismatch: sample is {sample.dtype}, "
            f"but conv_in expects {self.conv_in.weight.dtype}"
        )
        sample = self.conv_in(sample)

        # Create image-only indicator (all zeros for frame-conditioned model)
        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        down_block_res_samples = (sample,)

        # 3. Downsampling blocks
        if self.training and self.gradient_checkpointing:
            # Use gradient checkpointing to save memory during training
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):

                for downsample_block in self.down_blocks:
                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    else:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    down_block_res_samples += res_samples

                # 4. mid
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    emb,
                    encoder_hidden_states,
                    image_only_indicator,
                    use_reentrant=False,
                )

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[
                        : -len(upsample_block.resnets)
                    ]

                    if (
                        hasattr(upsample_block, "has_cross_attention")
                        and upsample_block.has_cross_attention
                    ):
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    else:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            image_only_indicator,
                            use_reentrant=False,
                        )
            else:

                for downsample_block in self.down_blocks:
                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                        )
                    else:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            image_only_indicator,
                        )
                    down_block_res_samples += res_samples

                # 4. mid
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    emb,
                    encoder_hidden_states,
                    image_only_indicator,
                )

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[
                        : -len(upsample_block.resnets)
                    ]

                    if (
                        hasattr(upsample_block, "has_cross_attention")
                        and upsample_block.has_cross_attention
                    ):
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                        )
                    else:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            image_only_indicator,
                        )

        else:
            for downsample_block in self.down_blocks:
                if (
                    hasattr(downsample_block, "has_cross_attention")
                    and downsample_block.has_cross_attention
                ):
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )

                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )

                down_block_res_samples += res_samples

            # 4. mid
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample = upsample_block(
                        hidden_states=sample,
                        res_hidden_states_tuple=res_samples,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        res_hidden_states_tuple=res_samples,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )

        # 6. Post-process: final convolution and normalization
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original frame dimensions
        # [batch * frames, channels, height, width] ->
        # [batch, frames, channels, height, width]
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        # Return structured output
        return UNetSpatioTemporalConditionOutput(sample=sample)
