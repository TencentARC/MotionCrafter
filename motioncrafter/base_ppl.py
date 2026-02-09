"""Base pipeline for MotionCrafter inference with shared encoding and decoding utilities."""

from typing import Optional

import gc
import torch
import torch.nn.functional as F
import numpy as np

# Import core components from diffusers library
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipeline,
)
from diffusers.utils import logging


# Get logger for this module
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MotionCrafterBasePipeline(StableVideoDiffusionPipeline):
    """
    Base pipeline for MotionCrafter providing shared encoding/decoding methods.
    
    Provides common functionality for video encoding, VAE encoding, and point map
    decoding that is used by both deterministic and diffusion-based pipelines.
    """

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        Encode video frames to embeddings using feature extractor.

        Args:
            video: Tensor of shape (B, C, H, W) in range [-1, 1].
            chunk_size: Number of frames to process at once.

        Returns:
            embeddings: Tensor of shape (T, 1024) containing image embeddings.
        """
        # Resize video to 224x224 with antialiasing
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        # Normalize from [-1, 1] to [0, 1]
        video_224 = (video_224 + 1.0) / 2.0
        embeddings = []
        # Process frames in chunks to manage memory
        for i in range(0, video_224.shape[0], chunk_size):
            emb = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            # Encode embeddings through image encoder
            embeddings.append(self.image_encoder(emb).image_embeds)  # [B, 1024]

        # Concatenate all frame embeddings
        embeddings = torch.cat(embeddings, dim=0)  # [T, 1024]
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        Encode video frames to VAE latent space.

        Args:
            video: Tensor of shape (B, C, H, W) in range [-1, 1].
            chunk_size: Number of frames to process at once.

        Returns:
            video_latents: Tensor of shape (T, C, H//8, W//8) containing VAE latents.
        """
        video_latents = []
        # Process frames in chunks for memory efficiency
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        # Concatenate all latents
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    @torch.no_grad()
    def decode_point_map(
        self,
        geometry_motion_vae,
        latents,
        chunk_size: int = 8,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        need_resize: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        low_memory_usage: bool = False,
    ):
        """
        Decode VAE latents to point maps, valid masks, and optionally deformation data.

        Args:
            geometry_motion_vae: VAE model for decoding point maps.
            latents: Tensor of shape (T, C, H, W) containing VAE latents.
            chunk_size: Number of frames to decode at once.
            force_projection: Whether to enforce camera projection constraints.
            force_fixed_focal: Whether to force fixed focal length.
            use_extract_interp: Use nearest-exact interpolation if True.
            need_resize: Whether to resize output to original resolution.
            height: Target height for resizing.
            width: Target width for resizing.
            low_memory_usage: Move intermediate results to CPU if True.

        Returns:
            Tuple containing:
            - rec_point_maps: (T, H, W, 3) point map tensor
            - rec_valid_masks: (T, H, W) boolean validity mask
            - rec_deformmaps: (T, H, W, 3) deformation map if available
            - rec_deformmasks: (T, H, W) deformation mask if available
        """
        T = latents.shape[0]
        rec_point_maps = []
        rec_valid_masks = []
        rec_deformmaps = []
        rec_deformmasks = []

        # Process latents in chunks
        for i in range(0, T, chunk_size):
            lat = latents[i : i + chunk_size]
            lat_1 = lat[:, :4, :, :]  # Point map latent channels

            # Decode point map based on VAE type
            if geometry_motion_vae.__class__.__name__ == "AutoencoderKL":
                rec_pointmap = geometry_motion_vae.decode(lat_1).sample
                rec_vmask = torch.ones_like(rec_pointmap[:, :1, :, :])
            else:
                rec_pointmap = geometry_motion_vae.decode(lat_1)
                rec_vmask = torch.ones_like(rec_pointmap[:, :1, :, :])

            # Store with optional CPU offloading for memory efficiency
            rec_point_maps.append(
                rec_pointmap.cpu() if low_memory_usage else rec_pointmap
            )
            rec_valid_masks.append(
                rec_vmask.cpu() if low_memory_usage else rec_vmask
            )

            # Decode deformation map if available in latents
            if lat.shape[1] > 4:
                lat_2 = lat[:, 4:8, :, :]
                if geometry_motion_vae.__class__.__name__ == "AutoencoderKL":
                    # Use standard VAE decoder
                    rec_deformmap = self.vae.decode(
                        lat_2,
                        num_frames=lat_2.shape[0],
                    ).sample
                    rec_deformmask = torch.ones_like(rec_deformmap[:, :1, :, :])
                elif geometry_motion_vae.__class__.__name__ == "SeperateAutoencoderKL":
                    # Use separate decoder branch
                    rec_deformmap = geometry_motion_vae.decode_2(lat_2)
                    rec_deformmask = torch.ones_like(rec_deformmap[:, :1, :, :])
                elif geometry_motion_vae.__class__.__name__ == "UnifyAutoencoderKL":
                    # Use unified decoder with latent 1 conditioning
                    rec_deformmap = geometry_motion_vae.decode_2(
                        lat_2,
                        latent_1=lat_1
                    )
                    rec_deformmask = torch.ones_like(rec_deformmap[:, :1, :, :])

                rec_deformmaps.append(
                    rec_deformmap.cpu() if low_memory_usage else rec_deformmap
                )
                rec_deformmasks.append(
                    rec_deformmask.cpu() if low_memory_usage else rec_deformmask
                )

        # Concatenate all chunks
        rec_point_maps = torch.cat(rec_point_maps, dim=0)
        rec_valid_masks = torch.cat(rec_valid_masks, dim=0)
        if len(rec_deformmaps) > 0:
            rec_deformmaps = torch.cat(rec_deformmaps, dim=0)
            rec_deformmasks = torch.cat(rec_deformmasks, dim=0)

        # Resize to original resolution if needed
        if need_resize:
            if use_extract_interp:
                # Use nearest-exact for discrete coordinates
                rec_point_maps = F.interpolate(
                    rec_point_maps, (height, width), mode='nearest-exact'
                )
                rec_valid_masks = F.interpolate(
                    rec_valid_masks, (height, width), mode='nearest-exact'
                )
            else:
                # Use bilinear interpolation for smooth surfaces
                rec_point_maps = F.interpolate(
                    rec_point_maps, (height, width), mode='bilinear',
                    align_corners=False
                )
                rec_valid_masks = F.interpolate(
                    rec_valid_masks, (height, width), mode='bilinear',
                    align_corners=False
                )

            if len(rec_deformmaps) > 0:
                if use_extract_interp:
                    rec_deformmaps = F.interpolate(
                        rec_deformmaps, (height, width), mode='nearest-exact'
                    )
                    rec_deformmasks = F.interpolate(
                        rec_deformmasks, (height, width), mode='nearest-exact'
                    )
                else:
                    rec_deformmaps = F.interpolate(
                        rec_deformmaps, (height, width), mode='bilinear',
                        align_corners=False
                    )
                    rec_deformmasks = F.interpolate(
                        rec_deformmasks, (height, width), mode='bilinear',
                        align_corners=False
                    )

        # Transpose from channels-first to channels-last format (T, H, W, 3)
        rec_point_maps = rec_point_maps.permute(0, 2, 3, 1)
        rec_valid_masks = rec_valid_masks.squeeze(1) > 0

        if len(rec_deformmaps) > 0:
            rec_deformmaps = rec_deformmaps.permute(0, 2, 3, 1)
            rec_deformmasks = rec_deformmasks.squeeze(1) > 0
            # Return both point maps and deformation maps
            return rec_point_maps, rec_valid_masks, rec_deformmaps, rec_deformmasks
        else:
            # Return only point maps
            return rec_point_maps, rec_valid_masks

    def _preprocess_video(
        self, video, height, width, decode_chunk_size, track_time=False
    ):
        """
        Preprocess video for inference.
        
        Returns:
            Tuple of (video_embeddings, video_latents, need_resize, 
                     original_height, original_width, num_frames, stride,
                     added_time_ids, device)
        """
        import gc
        
        # Convert numpy to tensor
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        
        height = height or video.shape[-2]
        width = width or video.shape[-1]
        original_height = video.shape[-2]
        original_width = video.shape[-1]
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        
        # Check inputs
        assert height % 64 == 0 and width % 64 == 0
        need_resize = original_height != height or original_width != width

        # Get device
        device = self._execution_device
        
        # Resize if needed
        if need_resize:
            video = F.interpolate(
                video, (height, width), mode="bicubic",
                align_corners=False, antialias=True
            ).clamp(0, 1)
        
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1]

        # Encode video embeddings
        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(0)

        # Encode VAE latents
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(0).to(video_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Prepare time embeddings
        added_time_ids = self._get_add_time_ids(
            7, 127, 0.02,
            video_embeddings.dtype,
            1, 1, False,
        ).to(device)

        if track_time:
            gc.collect()
            torch.cuda.empty_cache()

        return (
            video_embeddings, video_latents, need_resize,
            original_height, original_width, num_frames,
            added_time_ids, device
        )

    def _postprocess_latents(
        self, latents_all, geometry_motion_vae, window_size, num_frames,
        overlap, decode_chunk_size, force_projection, force_fixed_focal,
        use_extract_interp, need_resize, original_height, original_width,
        low_memory_usage=False, track_time=False
    ):
        """
        Postprocess and decode latents to point maps.
        
        Returns:
            Tuple of point maps, valid masks, and optionally deform maps/masks
        """
        import gc
        
        latents_all = 1 / self.vae.config.scaling_factor * latents_all.squeeze(0).to(
            geometry_motion_vae.dtype
        )

        if low_memory_usage:
            torch.cuda.empty_cache()

        # Decode point map
        if latents_all.shape[0] > 4:
            results = self.decode_point_map(
                geometry_motion_vae,
                latents_all,
                chunk_size=decode_chunk_size,
                force_projection=force_projection,
                force_fixed_focal=force_fixed_focal,
                use_extract_interp=use_extract_interp,
                need_resize=need_resize,
                height=original_height,
                width=original_width,
                low_memory_usage=low_memory_usage
            )
        else:
            results = self.decode_point_map(
                geometry_motion_vae,
                latents_all,
                chunk_size=decode_chunk_size,
                force_projection=force_projection,
                force_fixed_focal=force_fixed_focal,
                use_extract_interp=use_extract_interp,
                need_resize=need_resize,
                height=original_height,
                width=original_width,
                low_memory_usage=low_memory_usage
            )

        if track_time:
            gc.collect()
            torch.cuda.empty_cache()

        self.maybe_free_model_hooks()
        return results

    def _get_window_stride(self, num_frames, window_size, overlap):
        """
        Calculate window stride and update parameters if needed.
        
        Returns:
            Tuple of (window_size, overlap, stride)
        """
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap
        return window_size, overlap, stride

    def _process_windows(
        self,
        num_frames,
        window_size,
        overlap,
        video_latents,
        video_embeddings,
        added_time_ids,
        device,
        inference_fn,
    ):
        """
        Process video in sliding windows and accumulate latents.
        
        Args:
            num_frames: Total number of frames
            window_size: Size of processing window
            overlap: Overlap between windows
            video_latents: Encoded video latents (1, T, C, H, W)
            video_embeddings: Encoded video embeddings (1, T, 1024)
            added_time_ids: Time embeddings
            device: Computation device
            inference_fn: Callable that takes (video_latents_current, 
                         video_embeddings_current, added_time_ids) and 
                         returns latents for current window
        
        Returns:
            Accumulated latents from all windows (1, T, C, H, W)
        """
        latents_all = None
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        idx_start = 0
        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            
            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            # Run inference for this window
            latents = inference_fn(
                video_latents_current,
                video_embeddings_current,
                added_time_ids,
            )

            # Accumulate latents with overlap blending
            if latents_all is None:
                latents_all = latents.clone()
            else:
                if overlap > 0:
                    latents_all[:, -overlap:] = (
                        latents[:, :overlap] * weights +
                        latents_all[:, -overlap:] * (1 - weights)
                    )
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += window_size - overlap

        return latents_all
