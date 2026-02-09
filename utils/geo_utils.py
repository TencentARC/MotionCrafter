from typing import *

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from kornia.utils import create_meshgrid
from kornia.geometry.depth import depth_to_3d_v2


# Recover scale factor between predicted and ground truth point maps using least squares
@torch.no_grad()
def recover_scale(
    points: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor = None,
    weight: torch.Tensor = None,
    downsample_size: Tuple[int, int] = None
):
    """
    Recover the scale factor for a point map with a target point map by minimizing the mse loss.
    points * scale ~ points_gt

    ### Parameters:
    - `points: torch.Tensor` of shape (B, T, H, W, 3)
    - `points_gt: torch.Tensor` of shape (B, T, H, W, 3)
    - `mask: torch.Tensor` of shape (B, T, H, W) Optional.
    - `weight: torch.Tensor` of shape (B, T, H, W) Optional.
        - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map.
            Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `scale`: torch.Tensor of shape (B, ) the estimated scale factor
    """
    dtype = points.dtype
    device = points.device
    batch_size, num_frames, height, width, _ = points.shape
    # Downsample point maps for computational efficiency if specified
    if downsample_size is not None:
        points = rearrange(
            F.interpolate(rearrange(points, "b t h w c -> (b t) c h w"), downsample_size, mode='nearest'),
            "(b t) c h w -> b t h w c", b=batch_size
        )
        points_gt = rearrange(
            F.interpolate(rearrange(points_gt, "b t h w c -> (b t) c h w"), downsample_size, mode='nearest'),
            "(b t) c h w -> b t h w c", b=batch_size
        )
        mask = None if mask is None else rearrange(
            F.interpolate(
                rearrange(mask.to(torch.float32), "b t h w -> (b t) 1 h w"),
                downsample_size, mode='nearest'
            ),
            '(b t) 1 h w -> b t h w', b=batch_size
        ) > 0
        weight = None if weight is None else rearrange(
            F.interpolate(rearrange(weight, "b t h w -> (b t) 1 h w"), downsample_size, mode='nearest'),
            '(b t) 1 h w -> b t h w', b=batch_size
        )
    
    # Compute scale factor for each sample in the batch
    scale = []
    for i in range(batch_size):
        # Flatten point maps to shape (N, 3)
        points_ = points[i].reshape(-1, 3)
        points_gt_ = points_gt[i].reshape(-1, 3)
        mask_ = None if mask is None else mask[i].reshape(-1)
        weight_ = None if weight is None else weight[i].reshape(-1)
        # Skip if insufficient valid points
        if mask_ is not None and mask_.sum() < 8:
            scale.append(1.0) 
        # Apply mask to filter valid points
        if mask_ is not None:
            points_ = points_[mask_]
            points_gt_ = points_gt_[mask_]
            weight_ = None if weight is None else weight_[mask_]
        # Solve weighted least squares: min_x ||Ax-b||_2^2 where A=points, b=points_gt, x=scale
        A = points_.reshape(-1, 1)
        b = points_gt_.reshape(-1, 1)
        if weight_ is not None:
            weight_ = weight_.reshape(-1, 1).repeat(1, 3).reshape(-1)
            A = A * weight_[:, None]
            b = b * weight_[:, None]
        # Solve using numpy's least squares (supports weighted least squares)
        x = np.linalg.lstsq(A.cpu().numpy(), b.cpu().numpy(), rcond=None)[0]
        scale.append(x[0, 0])
    # Convert scale factors back to torch tensor
    scale = torch.tensor(scale, dtype=dtype, device=device)
    return scale

# Normalize point map by mean valid depth to ensure scale consistency
@torch.no_grad()
def normalize_point_map(point_map, valid_mask):
    # Input shapes: point_map (T,H,W,3), valid_mask (T,H,W)
    # Extract depth (Z-coordinate) and compute weighted mean using valid mask
    norm_factor = (point_map[..., 2] * valid_mask.float()).mean() / (valid_mask.float().mean() + 1e-8)
    # Clip normalization factor to avoid numerical instability
    norm_factor = norm_factor.clip(min=1e-3)
    # Normalize all coordinates by the depth mean
    return point_map / norm_factor

# Convert point map (X, Y) to intrinsic camera parameters (principal point, focal length)
@torch.no_grad()
def point_map_xy2intrinsic_map(point_map_xy):
    # Input shape: (*,h,w,2)
    height, width = point_map_xy.shape[-3], point_map_xy.shape[-2]
    # Ensure dimensions are even for proper mesh grid generation
    assert height % 2 == 0
    assert width % 2 == 0
    # Create normalized coordinate mesh grid in range [-1, 1]
    mesh_grid = create_meshgrid(
        height=height,
        width=width,
        normalized_coordinates=True,
        device=point_map_xy.device,
        dtype=point_map_xy.dtype
    )[0]  # Shape: (h,w,2)
    # Ensure mesh grid values are not too close to zero for numerical stability
    assert mesh_grid.abs().min() > 1e-4
    # Expand mesh grid to match input shape (*,h,w,2)
    mesh_grid = mesh_grid.expand_as(point_map_xy)
    # Compute principal point (nc) by averaging XY coordinates
    nc = point_map_xy.mean(dim=-2).mean(dim=-2)  # Shape: (*, 2)
    nc_map = nc[..., None, None, :].expand_as(point_map_xy)
    # Compute normalized focal length (nf) using: point_xy = mesh_grid * nf + nc
    nf = ((point_map_xy - nc_map) / mesh_grid).mean(dim=-2).mean(dim=-2)
    nf_map = nf[..., None, None, :].expand_as(point_map_xy)
    # Concatenate principal point and focal length maps
    return torch.cat([nc_map, nf_map], dim=-1)

# Convert depth map to 3D point map using camera intrinsic parameters
@torch.no_grad()
def depth2point_map(depth, fx, fy, cx, cy):
    # Input: depth of shape (h,w)
    assert len(depth.shape) == 2
    # Construct camera intrinsic matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ], dtype=fx.dtype, device=fx.device)
    # Back-project depth to 3D point cloud using: point = K^-1 * [x, y, 1]^T * depth
    point_maps = depth_to_3d_v2(
        depth.unsqueeze(0),
        camera_matrix.unsqueeze(0),
        normalize_points=False
    ).squeeze(0)  # Output shape: (h,w,3)
    return point_maps

# Compute robust min/max values using quantiles to avoid outliers
@torch.no_grad()
def robust_min_max(tensor, quantile=0.99):
    # Input shape: (T, H, W)
    T, H, W = tensor.shape
    min_vals = []
    max_vals = []
    # Compute quantile-based min/max for each frame to exclude outliers
    for i in range(T):
        # Use 1st and 99th percentile by default
        min_vals.append(torch.quantile(tensor[i], q=1-quantile, interpolation='nearest').item())
        max_vals.append(torch.quantile(tensor[i], q=quantile, interpolation='nearest').item())
    # Return global min and max across all frames
    return min(min_vals), max(max_vals)