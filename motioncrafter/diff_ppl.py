"""Diffusion-based motion generation pipeline."""

from typing import Callable, Dict, List, Optional, Union

import torch

# Import core components from diffusers library
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    retrieve_timesteps,
)
from .base_ppl import MotionCrafterBasePipeline


class MotionCrafterDiffPipeline(MotionCrafterBasePipeline):
    """
    Diffusion-based pipeline for motion generation based on SVD.
    
    Inherits encoding/decoding methods from base pipeline and implements
    multi-step iterative denoising with classifier-free guidance.
    """

    def _inference_step(
        self,
        video_latents_current,
        video_embeddings_current,
        added_time_ids,
        timesteps,
        num_warmup_steps,
        latents_all,
        device,
        generator,
        window_size,
        overlap,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
    ):
        """
        Perform multi-step diffusion inference for a single window.
        
        Returns:
            Latents for the current window
        """
        num_channels_latents = 16
        latents = self.prepare_latents(
            1,
            video_latents_current.shape[1],
            num_channels_latents,
            video_latents_current.shape[-2] * 8,
            video_latents_current.shape[-1] * 8,
            video_embeddings_current.dtype,
            device,
            generator,
            None,
        )

        weights = None
        if overlap > 0 and latents_all is not None:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # Blend with previous window overlap
                if latents_all is not None and i == 0 and overlap > 0:
                    latents[:, :overlap] = (
                        latents_all[:, -overlap:] +
                        latents[:, :overlap] / self.scheduler.init_noise_sigma *
                        self.scheduler.sigmas[i]
                    )

                # Predict noise without guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current], dim=2
                )
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if self.do_classifier_free_guidance:
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    latent_model_input = torch.cat(
                        [
                            latent_model_input,
                            torch.zeros_like(latent_model_input),
                            torch.zeros_like(latent_model_input),
                        ],
                        dim=2,
                    )
                    noise_pred_uncond = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=torch.zeros_like(
                            video_embeddings_current
                        ),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )

                # Step scheduler
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents

    @torch.no_grad()
    def __call__(
        self,
        video: Union[torch.Tensor],
        geometry_motion_vae,
        prior_model,
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        window_size: Optional[int] = 14,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        overlap: int = 4,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        track_time: bool = False,
        low_memory_usage: bool = False
    ):
        """
        Multi-step diffusion-based motion generation with CFG.
        
        Args:
            video: Input video [T, C, H, W] or [T, H, W, C]
            geometry_motion_vae: VAE for point map encoding/decoding
            prior_model: Prior model (unused in diffusion mode)
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            window_size: Processing window size
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Chunk size for decoding
            generator: Random generator
            latents: Initial latents (optional)
            callback_on_step_end: Callback after each step
            callback_on_step_end_tensor_inputs: Tensor inputs for callback
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

        # Set guidance scale
        self._guidance_scale = guidance_scale

        # Prepare timesteps for diffusion
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Define inference function for window processing
        def inference_fn(video_latents_curr, video_embeddings_curr, time_ids):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            return self._inference_step(
                video_latents_curr,
                video_embeddings_curr,
                time_ids,
                timesteps,
                num_warmup_steps,
                None,
                device,
                generator,
                window_size,
                overlap,
                callback_on_step_end,
                callback_on_step_end_tensor_inputs,
            )

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
