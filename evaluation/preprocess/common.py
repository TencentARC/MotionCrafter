import os

import h5py
import numpy as np
import torch
from kornia.geometry.depth import depth_to_3d_v2


def resolve_device(device_name: str) -> torch.device:
    # Keep device selection centralized so all preprocess scripts behave consistently.
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please set --device cpu or --device auto")
    return torch.device(device_name)


def depth_to_point_map(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, device: torch.device) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H, W], got shape={depth.shape}")

    # Convert depth + intrinsics into camera-space XYZ map.
    depth_tensor = torch.from_numpy(depth).float().to(device)
    camera_matrix = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=depth_tensor.dtype,
        device=device,
    )
    point_map = depth_to_3d_v2(
        depth_tensor.unsqueeze(0),
        camera_matrix.unsqueeze(0),
        normalize_points=False,
    ).squeeze(0)
    return point_map.cpu().numpy()


def center_crop(arr: np.ndarray, cropped_size):
    if arr.ndim < 3:
        raise ValueError(f"arr must have at least 3 dims [T, H, W, ...], got shape={arr.shape}")

    height, width = arr.shape[1], arr.shape[2]
    cropped_height, cropped_width = cropped_size

    top = max((height - cropped_height) // 2, 0)
    left = max((width - cropped_width) // 2, 0)
    end_h = min(top + cropped_height, height)
    end_w = min(left + cropped_width, width)
    # Works for both [T,H,W] and [T,H,W,C] style arrays.
    return arr[:, top:end_h, left:end_w, ...]


def normalize_rgb_frame(img: np.ndarray) -> np.ndarray:
    # Normalize to uint8 RGB for stable video encoding.
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    else:
        img = img[..., :3]
    return img.astype(np.uint8)


def write_hdf5(data_save_path: str, disps, valid_masks, point_maps, camera_poses):
    # Save with chunking on temporal dimension for efficient per-frame reads.
    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
    with h5py.File(data_save_path, "w") as h5f:
        h5f.create_dataset("disparity", data=disps.astype(np.float16), chunks=(1,) + disps.shape[1:])
        h5f.create_dataset("valid_mask", data=valid_masks.astype(np.bool_), chunks=(1,) + valid_masks.shape[1:])
        h5f.create_dataset("point_map", data=point_maps.astype(np.float16), chunks=(1,) + point_maps.shape[1:])
        h5f.create_dataset("camera_pose", data=camera_poses.astype(np.float16), chunks=(1,) + camera_poses.shape[1:])