"""Loss utilities for geometry and depth tasks with optional weighting."""

import torch.nn.functional as F
import torch
from einops import rearrange
from kornia.filters import spatial_gradient
import kornia.core as kornia_ops


# Mean-squared error with distance-aware weights to down-weight far points
def distance_weighted_mse_loss(input, target, weight=None):
    """Compute MSE; if weight is provided, reweight by inverse distance to the target."""
    if weight is not None:
        loss = F.mse_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1)
        distance = (target ** 2).sum(dim=-1).sqrt()
        distance_mean = (distance * weight).mean() / weight.mean().add(1e-6)
        distance = distance / (distance_mean + 1e-6)
        weight = weight * (1.0 / (distance + 1e-6))
        loss = (loss * weight).mean() / weight.mean().add(1e-7)
        return loss        
    else:
        return F.mse_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1).mean()


# L1 loss with optional sample-wise weights
def weighted_l1_loss(input, target, weight=None):
    """Compute L1 loss; if weight is provided, average using the given mask weights."""
    if weight is not None:
        loss = F.l1_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1)
        loss = (loss * weight).mean() / weight.mean().add(1e-7)
        return loss        
    else:
        return F.l1_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1).mean()


# MSE loss with optional sample-wise weights
def weighted_mse_loss(input, target, weight=None):
    """Compute MSE; if weight is provided, average using the given mask weights."""
    if weight is not None:
        loss = F.mse_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1)
        loss = (loss * weight).mean() / weight.mean().add(1e-7)
        return loss        
    else:
        return F.mse_loss(
            input,
            target, 
            reduction='none'
        ).sum(dim=-1).mean()


# Normal consistency loss from point maps; uses spatial gradients to compute normals
def weighted_normal_loss(input, target, weight=None):
    """Compute cosine loss between predicted and target normals with optional mask weights."""
    # Input shapes: input/target (B,T,H,W,3), weight (B,T,H,W)

    def compute_normal(pmap):
        """Compute normals via cross product of spatial gradients on point maps."""
        gradients = spatial_gradient(rearrange(pmap, "b h w c -> b c h w"))  # Bx3x2xHxW
        gradients = gradients[:, :, :, 1:-1, 1:-1]  # drop edge pixels
        a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xH-2xW-2
        normals = torch.cross(a, b, dim=1)  # Bx3xH-2xW-2
        normals = kornia_ops.normalize(normals, dim=1, p=2, eps=1e-6)
        return rearrange(normals, "b c h w -> b h w c")
    
    inp_norm = compute_normal(rearrange(input, "b t h w c -> (b t) h w c"))
    tgt_norm = compute_normal(rearrange(target, "b t h w c -> (b t) h w c"))
    
    if weight is not None:
        # Smooth weight mask with a 3x3 averaging kernel to ignore thin holes
        kernel = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            device=weight.device,
            dtype=weight.dtype
        )
        norm = kernel.abs().sum(dim=-1).sum(dim=-1)
        kernel = kernel / (norm[..., None, None])
        # 3*3
        B, T, H, W = weight.shape
        weight = F.conv2d(
            weight.reshape(-1, 1, H, W),
            kernel[None, None],
            groups=1,
            padding=0,
            stride=1
        )
        weight = weight.reshape(B, T, H-2, W-2)
        weight = (weight > 0.99).float()

        loss = (1.0 - torch.sum(
            inp_norm * tgt_norm, dim=-1  
        )) * rearrange(weight, "b t h w -> (b t) h w")
        loss = loss.mean() / weight.mean().add(1e-7)
        return loss        
    else:
        return (1.0 - torch.sum(
            inp_norm * tgt_norm, dim=-1  
        )).mean()


# Multi-scale depth loss that aggregates patch-wise weighted L1 at several spatial scales
def multi_scale_weighted_depth_loss(input, target, weight, scale_factors=[4, 16, 64]):
    """Compute weighted depth loss across multiple patch scales for robustness."""
    # Inputs: input/target/weight (B,T,H,W)

    inp = rearrange(input, "b t h w -> (b t) h w")
    tgt = rearrange(target, "b t h w -> (b t) h w") 
    wgt = rearrange(weight, "b t h w -> (b t) h w")
    H, W = inp.shape[-2], inp.shape[-1]
    loss = 0
    for scale_factor in scale_factors:
        assert H % scale_factor == 0
        assert W % scale_factor == 0
        patch_H, patch_W = H // scale_factor, W // scale_factor
        inp_patch = inp.reshape(
            -1, scale_factor, patch_H, scale_factor, patch_W
        ).permute(0, 1, 3, 2, 4).reshape(-1, patch_H, patch_W)
        tgt_patch = tgt.reshape(
            -1, scale_factor, patch_H, scale_factor, patch_W
        ).permute(0, 1, 3, 2, 4).reshape(-1, patch_H, patch_W)
        weight_patch = wgt.reshape(
            -1, scale_factor, patch_H, scale_factor, patch_W
        ).permute(0, 1, 3, 2, 4).reshape(-1, patch_H, patch_W)
        inp_mean_patch = torch.mean(inp_patch * weight_patch, dim=[-2, -1], keepdim=True) / (
            torch.mean(weight_patch, dim=[-2, -1], keepdim=True) + 1e-6
        )
        tgt_mean_patch = torch.mean(tgt_patch * weight_patch, dim=[-2, -1], keepdim=True) / (
            torch.mean(weight_patch, dim=[-2, -1], keepdim=True) + 1e-6
        )
        inp_patch = inp_patch - inp_mean_patch.detach()
        tgt_patch = tgt_patch - tgt_mean_patch.detach()
        loss += weighted_l1_loss(inp_patch.unsqueeze(-1), tgt_patch.unsqueeze(-1), weight_patch)
    
    return loss


# Chamfer distance between two point clouds with optional masks
def chamfer_distance(pcd1, pcd2, mask1=None, mask2=None):
    """Compute bidirectional Chamfer distance for batched point maps."""
    # Inputs: pcd1/pcd2 (B,T,H,W,C), mask1/mask2 (B,T,H,W)
    B, T, H, W, C = pcd1.shape
    if mask1 is not None:
        pcd1 = pcd1[mask1]
    if mask2 is not None:
        pcd2 = pcd2[mask2]
    pcd1 = pcd1.reshape(B*T, -1, C)
    pcd2 = pcd2.reshape(B*T, -1, C)
    diff = pcd1[:, :, None, :] - pcd2[:, None, :, :]  # (B*T, N, M, C)
    dist = torch.sum(diff ** 2, dim=-1)  # (B*T, N, M)
    dist1 = torch.min(dist, dim=-1)[0]  # (B*T, N)
    dist2 = torch.min(dist, dim=-2)[0]  # (B*T, M)
    cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)).mean()
    return cd