"""Shared helpers for dataset preprocessing scripts."""

import os
from typing import Any, Dict, Iterable, List
import torch
from kornia.geometry.depth import depth_to_3d_v2


def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def get_env_list(name: str, default_csv: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, default_csv).split(",") if item.strip()]


def write_meta_infos(output_dir: str, meta_infos: Iterable[Dict[str, Any]]) -> None:
    with open(os.path.join(output_dir, "meta_infos.txt"), "w", encoding="utf-8") as file:
        for meta in meta_infos:
            print(meta["video"], meta["data"], meta["num_frames"], file=file)


def depth2point_map(depth: torch.Tensor, fx, fy, cx, cy) -> torch.Tensor:
    """Convert depth maps to camera-space point maps.

    Supports both single-frame depth tensors with shape (H, W) and batched
    depth tensors with shape (T, H, W). Camera intrinsics can be either
    scalars (single frame) or per-frame vectors (batched mode).
    """
    if depth.ndim == 2:
        camera_matrix = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
            dtype=depth.dtype,
            device=depth.device,
        )
        return depth_to_3d_v2(
            depth.unsqueeze(0),
            camera_matrix.unsqueeze(0),
            normalize_points=False,
        ).squeeze(0)

    if depth.ndim == 3:
        fx_t = torch.as_tensor(fx, dtype=depth.dtype, device=depth.device)
        fy_t = torch.as_tensor(fy, dtype=depth.dtype, device=depth.device)
        cx_t = torch.as_tensor(cx, dtype=depth.dtype, device=depth.device)
        cy_t = torch.as_tensor(cy, dtype=depth.dtype, device=depth.device)

        zero = torch.zeros_like(fx_t)
        one = torch.ones_like(fx_t)
        intrinsics = torch.stack(
            [fx_t, zero, cx_t, zero, fy_t, cy_t, zero, zero, one],
            dim=-1,
        ).reshape(-1, 3, 3)
        return depth_to_3d_v2(depth, intrinsics, normalize_points=False).float()

    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")
