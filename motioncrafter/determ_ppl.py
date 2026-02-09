"""Deterministic motion generation pipeline."""

from typing import Optional, Union

import torch

from .base_ppl import MotionCrafterBasePipeline


class MotionCrafterDetermPipeline(MotionCrafterBasePipeline):
    """
    Deterministic pipeline for motion generation based on SVD.
    
    Inherits encoding/decoding methods from base pipeline and implements
    a single-step inference strategy without iterative denoising.
    """

    @torch.no_grad()
    def __call__(
        self,
        video: Union[torch.Tensor],
        geometry_motion_vae,
        prior_model,
        height: int = 576,
        width: int = 1024,
        window_size: Optional[int] = 14,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        overlap: int = 4,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        track_time: bool = False,
        low_memory_usage: bool = False,
        **kwargs
    ):
        """
        Single-step deterministic motion generation.
        
        Args:
            video: Input video [T, C, H, W] or [T, H, W, C]
            geometry_motion_vae: VAE for point map encoding/decoding
            prior_model: Prior model (unused in deterministic mode)
            height: Output height
            width: Output width
            window_size: Processing window size
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Chunk size for decoding
            overlap: Overlap between windows
            force_projection: Apply camera projection
            force_fixed_focal: Use fixed focal length
            use_extract_interp: Use interpolation for extraction
            track_time: Track inference timing
            low_memory_usage: Enable low memory mode
            
        Returns:
            Tuple of (point_map, valid_mask) or (point_map, valid_mask, 
                     deform_map, deform_mask)
        """
        # Preprocess video
        (video_embeddings, video_latents, need_resize,
         original_height, original_width, num_frames,
         added_time_ids, device) = self._preprocess_video(
            video, height, width, decode_chunk_size, track_time
        )

        # Get window parameters
        window_size, overlap, stride = self._get_window_stride(
            num_frames, window_size, overlap
        )

        # Fixed timestep for deterministic inference
        timestep = 1.6378
        self._num_timesteps = 1

        # Define inference function for single-step deterministic inference
        def inference_fn(video_latents_curr, video_embeddings_curr, time_ids):
            model_pred = self.unet(
                video_latents_curr,
                timestep,
                encoder_hidden_states=video_embeddings_curr,
                added_time_ids=time_ids,
                return_dict=False,
            )[0]
            return model_pred * -1

        # Process video in windows
        latents_all = self._process_windows(
            num_frames,
            window_size,
            overlap,
            video_latents,
            video_embeddings,
            added_time_ids,
            device,
            inference_fn,
        )

        # Postprocess and decode
        return self._postprocess_latents(
            latents_all, geometry_motion_vae, window_size, num_frames,
            overlap, decode_chunk_size, force_projection, force_fixed_focal,
            use_extract_interp, need_resize, original_height, original_width,
            low_memory_usage, track_time
        )
