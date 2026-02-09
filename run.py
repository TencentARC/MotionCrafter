"""Inference entrypoint for MotionCrafter pipelines (diffusion and deterministic)."""

from pathlib import Path
import torch
from decord import VideoReader, cpu
from diffusers.training_utils import set_seed
from fire import Fire
import os
import numpy as np
import torch.nn.functional as F
import h5py

from diffusers import AutoencoderKL
from motioncrafter import (
    MotionCrafterDiffPipeline,
    MotionCrafterDetermPipeline,
    UnifyAutoencoderKL,
    UNetSpatioTemporalConditionModelVid2vid
)


def main(
    video_path: str,
    num_frames: int = -1,
    save_folder: str = "workspace/output/",
    cache_dir: str = "workspace/cache", 
    unet_path: str = "TencentARC/MotionCrafter",
    vae_path: str = "TencentARC/MotionCrafter",
    height: int = None,
    width: int = None,
    downsample_ratio: float = 1.0,
    adjust_resolution: bool = False,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 25,
    sliding_window: bool = False,
    overlap: int = 5,
    decode_chunk_size: int = 25,
    process_length: int = -1,
    process_stride: int = 1,
    seed: int = 42,
    model_type: str = 'diff',
    force_projection: bool = True,
    force_fixed_focal: bool = True,
    use_extract_interp: bool = False,
    track_time: bool = False,
    low_memory_usage: bool = False
):
    """Run MotionCrafter inference on a video to generate point maps and scene flow.
    
    This function performs end-to-end inference using either diffusion-based or 
    deterministic motion generation models. It processes input videos to produce 
    3D point maps and optionally scene flow predictions.
    
    Args:
        video_path: Path to input video file (MP4 format recommended).
        num_frames: Number of frames to process from video. If -1, processes all frames.
        save_folder: Directory path to save output NPZ files containing predictions.
        cache_dir: Directory for caching downloaded models.
        unet_path: HuggingFace model ID or local path for UNet model.
        vae_path: HuggingFace model ID or local path for geometry motion VAE.
        height: Target height for processing. Must be divisible by 64. If None, uses original.
        width: Target width for processing. Must be divisible by 64. If None, uses original.
        downsample_ratio: Ratio to downsample video before processing (>1.0 reduces resolution).
        adjust_resolution: Whether to resize and center crop video to target resolution.
        num_inference_steps: Number of denoising steps for diffusion model (ignored for deterministic).
        guidance_scale: Classifier-free guidance scale for diffusion model (1.0 = no guidance).
        window_size: Number of frames to process in each temporal window.
        sliding_window: Whether to use sliding window inference (currently not implemented).
        overlap: Number of overlapping frames between consecutive windows.
        decode_chunk_size: Number of frames to decode at once from latent space.
        process_length: Maximum number of frames to process. If -1, processes all sampled frames.
        process_stride: Stride for frame sampling (1 = every frame, 2 = every other frame, etc.).
        seed: Random seed for reproducibility.
        model_type: Type of model to use - 'diff' for diffusion or 'determ' for deterministic.
        force_projection: Whether to enforce camera projection constraints during decoding.
        force_fixed_focal: Whether to use fixed focal length assumption.
        use_extract_interp: Whether to use nearest-exact interpolation for feature extraction.
        track_time: Whether to log timing information for profiling.
        low_memory_usage: Enable low memory mode by offloading intermediate results to CPU.
        
    Returns:
        None. Saves results to NPZ file containing:
            - point_map: (T, H, W, 3) array of 3D point coordinates
            - valid_mask: (T, H, W) boolean array indicating valid points
            - scene_flow: (T, H, W, 3) array of scene flow vectors (if available)
            - deform_mask: (T, H, W) boolean array for scene flow validity (if available)
            
    Raises:
        AssertionError: If height or width is not divisible by 64.
        AssertionError: If model_type is not 'diff' or 'determ'.
        NotImplementedError: If sliding_window is True (feature not yet implemented).
        
    Example:
        >>> main(
        ...     video_path="input.mp4",
        ...     model_type="diff",
        ...     num_inference_steps=25,
        ...     guidance_scale=1.5,
        ...     height=320,
        ...     width=576
        ... )
    """
    assert model_type in ['diff', 'determ'], f"model_type must be 'diff' or 'determ', got {model_type}"
    set_seed(seed)
    
    # Load UNet model for motion generation
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        unet_path,
        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float16)
    
    # Load geometry and motion VAE for point map decoding
    geometry_motion_vae = UnifyAutoencoderKL.from_pretrained(
        vae_path,
        subfolder='geometry_motion_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float32)
    
    # Initialize pipeline based on model type
    if model_type == 'diff':
        pipe = MotionCrafterDiffPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    else:
        pipe = MotionCrafterDetermPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    
    # Print model parameter counts for bookkeeping
    unet_params = sum(p.numel() for p in pipe.unet.parameters())
    pointmap_vae_params = sum(p.numel() for p in geometry_motion_vae.decoder.parameters())
    scene_flow_vae_params = sum(p.numel() for p in geometry_motion_vae.decoder_2.parameters()) 
    video_vae_params = sum(p.numel() for p in pipe.vae.encoder.parameters())
    total_params = unet_params + pointmap_vae_params + scene_flow_vae_params + video_vae_params
    print(f"Unet parameters: {unet_params/1e6:.2f}M")
    print(f"PointMap VAE decoder parameters: {(pointmap_vae_params + scene_flow_vae_params)/1e6:.2f}M")
    print(f"Video VAE encoder parameters: {video_vae_params/1e6:.2f}M")
    print(f"Total parameters: {total_params/1e6:.2f}M")

    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("Xformers is not enabled - falling back to standard attention")
    
    # Enable attention slicing to reduce memory usage
    # Note: Workaround for issue at https://github.com/continue-revolution/sd-webui-animatediff/issues/101
    pipe.enable_attention_slicing()
    
    # Load video and get dimensions
    video_base_name = os.path.basename(video_path).split('.')[0]
    vid = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    # Use original dimensions if not specified
    if height is None or width is None:
        height = original_height
        width = original_width
    
    # Validate dimensions are compatible with model architecture
    assert height % 64 == 0, f"Height {height} must be divisible by 64"
    assert width % 64 == 0, f"Width {width} must be divisible by 64"

    # Sample frames from video based on parameters
    if num_frames > 0:
        video_length = min(len(vid), num_frames)
    else:
        video_length = len(vid)
    frames_idx = list(range(0, video_length, process_stride))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
    
    # Limit processing length if specified
    if process_length > 0:
        process_length = min(process_length, len(frames))
        frames = frames[:process_length]
    else:
        process_length = len(frames)
    
    # Adjust window size and overlap based on sequence length
    window_size = min(window_size, process_length)
    if window_size == process_length: 
        overlap = 0
    
    # Convert frames to tensor and normalize to [0, 1]
    frames_tensor = torch.tensor(frames.astype("float32"), device='cuda').float().permute(0, 3, 1, 2)
    # Shape: (T, 3, H, W)

    if downsample_ratio > 1.0:
        # Store original dimensions for later upsampling
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        # Downsample frames to reduce computational cost
        frames_tensor = F.interpolate(
            frames_tensor,
            (
                round(frames_tensor.shape[-2] / downsample_ratio),
                round(frames_tensor.shape[-1] / downsample_ratio),
            ),
            mode='bicubic',
            antialias=True
        ).clamp(0, 1)

    # Create output directory
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    if adjust_resolution:
        # Store original dimensions before resizing
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        # Cover resize: scale to cover target resolution while maintaining aspect ratio
        r = max(height / frames_tensor.shape[-2], width / frames_tensor.shape[-1])
        new_size = int(frames_tensor.shape[-2] * r), int(frames_tensor.shape[-1] * r)
        frames_tensor = F.interpolate(frames_tensor, new_size, mode='bicubic', antialias=True).clamp(0, 1)
        # Center crop to exact target size
        h_start = (frames_tensor.shape[-2] - height) // 2
        w_start = (frames_tensor.shape[-1] - width) // 2
        frames_tensor = frames_tensor[:, :, h_start:h_start+height, w_start:w_start+width]

        # Save resized video for reference
        resized_frames = (frames_tensor.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        import imageio
        imageio.mimwrite(save_path / f"{video_base_name}.mp4", resized_frames, fps=30, quality=8)

    if sliding_window:
        raise NotImplementedError("Sliding window inference is not implemented yet. Use window_size parameter instead.")

    else:
        # Run inference on the full video sequence
        with torch.inference_mode():
            print("Running inference...")
            results = pipe(
                frames_tensor,
                geometry_motion_vae,
                None,  # prior_model not used
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                decode_chunk_size=decode_chunk_size,
                overlap=0,
                force_projection=force_projection,
                force_fixed_focal=force_fixed_focal,
                use_extract_interp=use_extract_interp,
                track_time=track_time,
                low_memory_usage=low_memory_usage
            )
            print("Inference completed.")

            # Unpack results based on model output
            if len(results) == 4:
                rec_point_map, rec_valid_mask, rec_deform_map, rec_deform_mask = results
            else:
                rec_point_map, rec_valid_mask = results
                rec_deform_map, rec_deform_mask = None, None

        if downsample_ratio > 1.0:
            # Upsample predictions back to original video resolution
            rec_point_map = F.interpolate(
                rec_point_map.permute(0, 3, 1, 2),
                (original_height, original_width),
                mode='bilinear'
            ).permute(0, 2, 3, 1)
            rec_valid_mask = F.interpolate(
                rec_valid_mask.float().unsqueeze(1),
                (original_height, original_width),
                mode='bilinear'
            ).squeeze(1) > 0.5
            if rec_deform_map is not None:
                rec_deform_map = F.interpolate(
                    rec_deform_map.permute(0, 3, 1, 2),
                    (original_height, original_width),
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
                rec_deform_mask = F.interpolate(
                    rec_deform_mask.float().unsqueeze(1),
                    (original_height, original_width),
                    mode='bilinear'
                ).squeeze(1) > 0.5

        # Save results to compressed NPZ format
        if rec_deform_map is None:
            # Save only point map and validity mask
            np.savez(
                str(save_path / f"{video_base_name}.npz"), 
                point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
                valid_mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_),
            )
        else:
            # Save point map, validity mask, scene flow, and deformation mask
            np.savez(
                str(save_path / f"{video_base_name}.npz"), 
                point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
                valid_mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_),
                scene_flow=rec_deform_map.detach().cpu().numpy().astype(np.float16),
                deform_mask=rec_deform_mask.detach().cpu().numpy().astype(np.bool_),
            )
        
        print(f"Saved results to {save_path / f'{video_base_name}.npz'}")

if __name__ == "__main__":
    Fire(main)
