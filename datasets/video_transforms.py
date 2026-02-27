"""Data augmentation and geometric transform utilities for video-based training."""

import random
from typing import Tuple, Union, Optional, List, Any, Dict

import torchvision
import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np

torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import (
    functional as F,
    InterpolationMode,
    ColorJitter as ColorJit,
    Compose
)

import torch.nn.functional as Fnn

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")

        super().__init__()
        self.p = p
    
    def forward(self, input_dict: Dict) -> Dict:
        if torch.rand(1) >= self.p:
            return input_dict
        
        frame = input_dict['frame']
        size = frame.shape[-2:]

        output_dict = dict()
        for k, v in input_dict.items():
            if v.shape[-2:] == size:
                output_dict[k] = F.horizontal_flip(v)
                if k == 'point_map':
                    output_dict[k][:, 0] = -output_dict[k][:, 0]
            else:
                output_dict[k] = v
        return output_dict

class CoverResize(nn.Module):
    def __init__(
        self,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, size, cropped_size):
        r = max(cropped_size[0] / size[0], cropped_size[1] / size[1])
        new_size = int(size[0] * r), int(size[1] * r)
        return new_size

    def forward(self, input_dict: Dict) -> Dict:

        frame = input_dict['frame']
        size = frame.shape[-2:]
        cropped_size = input_dict.get('resolution', None) if input_dict.get('resolution', None) is not None else size
        new_size = self._get_params(size, cropped_size)

        need_resize = new_size[0] != size[0] or new_size[1] != size[1]
        if need_resize:
            output_dict = dict()
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[-2:] == size:
                    output_dict[k] = F.resize(
                        v,
                        new_size,
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    )
                else:
                    output_dict[k] = v
        else:
            output_dict = input_dict
        return output_dict


class RandomResize(CoverResize):
    def __init__(
        self,
        size_ratio_limit: float=None,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = True,
    ):
        super().__init__(interpolation, antialias)
        self.size_ratio_limit = size_ratio_limit

    def _get_params(self, size, cropped_size):
        r = max(cropped_size[0] / size[0], cropped_size[1] / size[1])
        if self.size_ratio_limit is not None:
            assert self.size_ratio_limit >= 1.0
            r = random.uniform(r, self.size_ratio_limit * r)
        elif r <= 1.0:
            r = random.uniform(r, 1)
        else:
            pass # same as cover resize
        new_size = int(size[0] * r), int(size[1] * r)
        assert new_size[0] >= cropped_size[0] and new_size[1] >= cropped_size[1]
        return new_size
    
class BaseCrop(nn.Module):
    """Base class for crop operations with common logic."""
    
    def __init__(self):
        super().__init__()
    
    def _get_crop_position(self, height, width, cropped_height, cropped_width):
        """
        Calculate crop position. Must be implemented by subclasses.
        
        Returns:
            Tuple of (top, left, needs_crop)
        """
        raise NotImplementedError
    
    def _crop_tensors(self, input_dict, size, top, left, cropped_height, cropped_width):
        """Apply cropping to all tensors with matching spatial dimensions."""
        output_dict = dict()
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor) and v.shape[-2:] == size:
                output_dict[k] = F.crop(
                    v,
                    top=top,
                    left=left,
                    height=cropped_height,
                    width=cropped_width
                )
            else:
                output_dict[k] = v
        return output_dict
    
    def forward(self, input_dict: Dict) -> Dict:
        frame = input_dict['frame']
        size = frame.shape[-2:]
        cropped_size = (
            input_dict.get('resolution', None)
            if input_dict.get('resolution', None) is not None
            else size
        )
        height, width = size
        cropped_height, cropped_width = cropped_size
        
        # Get crop position from subclass
        top, left, needs_crop = self._get_crop_position(
            height, width, cropped_height, cropped_width
        )
        
        if needs_crop:
            return self._crop_tensors(
                input_dict, size, top, left, cropped_height, cropped_width
            )
        else:
            return input_dict


class RandomCrop(BaseCrop):
    """Random crop with random position."""
    
    def _get_crop_position(self, height, width, cropped_height, cropped_width):
        """Calculate random crop position."""
        needs_vert_crop, top = (
            (True, int(torch.randint(0, height - cropped_height + 1, size=())))
            if height > cropped_height
            else (False, 0)
        )
        needs_horz_crop, left = (
            (True, int(torch.randint(0, width - cropped_width + 1, size=())))
            if width > cropped_width
            else (False, 0)
        )
        needs_crop = needs_vert_crop or needs_horz_crop
        return top, left, needs_crop


class CenterCrop(BaseCrop):
    """Center crop with center position."""
    
    def _get_crop_position(self, height, width, cropped_height, cropped_width):
        """Calculate center crop position."""
        needs_vert_crop, top = (
            (True, (height - cropped_height) // 2)
            if height > cropped_height
            else (False, 0)
        )
        needs_horz_crop, left = (
            (True, (width - cropped_width) // 2)
            if width > cropped_width
            else (False, 0)
        )
        needs_crop = needs_vert_crop or needs_horz_crop
        return top, left, needs_crop

    
class ColorJitter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    
    def forward(self, input_dict: Dict) -> Dict:

        output_dict = input_dict
        frame = (input_dict['frame'] + 1) / 2 # [0, 1]
        jitter = ColorJit(**self.kwargs)
        output_dict['frame'] = jitter(frame) * 2 - 1 # [-1, 1]
        return output_dict
    

class PointMapNormalize(nn.Module):
    def __init__(self, is_normalized=False, cuboid=True):
        super().__init__()
        self.is_normalized = is_normalized
        self.cuboid = cuboid
    
    
    def normalize(self, point_map, valid_mask, camera_pose):
        # T,H,W,3 T,H,W
        norm_factor = (point_map[..., 2] * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        norm_factor = norm_factor.clip(min=1e-8)
        point_map = point_map / norm_factor
        camera_pose[:, :3, 3] = camera_pose[:, :3, 3] / norm_factor
        return point_map, camera_pose


    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        point_map = input_dict['point_map'].permute(0, 2, 3, 1)
        camera_pose = input_dict['camera_pose']
        if not self.is_normalized:
            point_map, camera_pose = self.normalize(point_map, input_dict['valid_mask'] > 0.5, camera_pose)
        if self.cuboid:
            point_map[..., :2] = point_map[..., :2] / (point_map[..., 2:3] + 1e-7)
            # [x/z, y/z, log(z)]
            point_map[..., 2] = (
                torch.log(point_map[..., 2] + 1e-7)
                * (input_dict['valid_mask'] > 0.5)
            )
        output_dict['point_map'] = point_map.permute(0, 3, 1, 2)
        return output_dict

class WorldMapNormalize(nn.Module):
    def __init__(self, rescale=True, reshift=True):
        super().__init__()
        self.rescale = rescale
        self.reshift = reshift

    def normalize(self, world_map, valid_mask, camera_pose, scene_flow=None):
        
        if self.reshift:
            # Translate point cloud to ensure centroid is at origin
            valid_points = world_map[valid_mask > 0.5]  # Select valid points
            # Calculate mean of valid points as centroid
            centroid = valid_points.mean(dim=0)
            world_map = world_map - centroid
            camera_pose[:, :3, 3] = camera_pose[:, :3, 3] - centroid.unsqueeze(0)

        if self.rescale:
            # Calculate max scale: mean euclidean distance from valid points
            max_distance = (world_map.norm(p=2, dim=-1)[valid_mask > 0.5]).mean()
            world_map = world_map / (max_distance + 1e-8)  # Prevent division by zero
            camera_pose[:, :3, 3] = camera_pose[:, :3, 3] / (max_distance + 1e-8)
            scene_flow = scene_flow / (max_distance + 1e-8) if scene_flow is not None else None
        
        scale_infos = {
            'shift': centroid if self.reshift else torch.zeros(3, device=world_map.device),
            'scale': max_distance if self.rescale else torch.ones(1, device=world_map.device)
        }
        return world_map, camera_pose, scene_flow, scale_infos

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        world_map = input_dict['point_map'].permute(0, 2, 3, 1)
        camera_pose = input_dict['camera_pose']
        if 'scene_flow' in input_dict and input_dict['scene_flow'] is not None:
            scene_flow = input_dict['scene_flow'].permute(0, 2, 3, 1)
        else:
            scene_flow = None
        world_map, camera_pose, scene_flow, scale_infos = self.normalize(
            world_map, input_dict['valid_mask'] > 0.5, camera_pose, scene_flow
        )
        output_dict['point_map'] = world_map.permute(0, 3, 1, 2)
        output_dict['camera_pose'] = camera_pose
        output_dict['scale_infos'] = scale_infos
        if scene_flow is not None:
            output_dict['scene_flow'] = scene_flow.permute(0, 3, 1, 2)
        return output_dict
    

class CameraPoseNormalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def normalize(self, camera_pose):
        if torch.isnan(camera_pose).any():
            raise ValueError("camera_pose contains NaN values before normalization.")        
        # camera_pose: T, 4, 4
        first_pose = deepcopy(camera_pose[0])  # 4, 4
        camera_pose[:, :3, :3] = first_pose[:3, :3].T @ camera_pose[:, :3, :3]
        camera_pose[:, :3, 3] = (first_pose[:3, :3].T @ (camera_pose[:, :3, 3] - first_pose[:3, 3]).T).T
        # if torch.isnan(camera_pose).any():
        #     raise ValueError("camera_pose contains NaN values after normalization.")
        return camera_pose

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        camera_pose = input_dict['camera_pose']
        camera_pose = self.normalize(camera_pose)

        output_dict['camera_pose'] = camera_pose
        return output_dict


class PointMaptoWorld(nn.Module):
    def __init__(self):
        super().__init__()

    def point_map_to_world(self, point_map, camera_pose):
        # point_map: T, H, W, 3
        # camera_pose: T, 4, 4
        batch_size, height, width, _ = point_map.shape
        point_map = point_map.reshape(batch_size, height * width, 3)
        point_map = point_map.permute(0, 2, 1)  # T, 3, H*W
        world_map = torch.bmm(camera_pose[:, :3, :3], point_map) + camera_pose[:, :3, 3].unsqueeze(-1)  # T, 3, H*W
        world_map = world_map.permute(0, 2, 1)  # T, H*W, 3
        world_map = world_map.reshape(batch_size, height, width, 3)
        return world_map

    def scene_flow_to_world(self, scene_flow, camera_pose, point_map, world_map):
        # scene_flow: T, H, W, 3
        # camera_pose: T, 4, 4
        # point_map: T, H, W, 3
        # world_map: T, H, W, 3
        batch_size, height, width, _ = point_map.shape
        scene_flow = scene_flow.reshape(batch_size, height * width, 3).permute(0, 2, 1) # T, 3, H*W
        point_map = point_map.reshape(batch_size, height * width, 3).permute(0, 2, 1) # T, 3, H*W
        world_map = world_map.reshape(batch_size, height * width, 3).permute(0, 2, 1) # T, 3, H*W

        point_map_deform = point_map[:-1] + scene_flow[:-1]  # T-1, 3, H*W
        camera_pose = camera_pose[1:] # T-1, 4, 4
        world_map_deform = (
            torch.bmm(camera_pose[:, :3, :3], point_map_deform)
            + camera_pose[:, :3, 3].unsqueeze(-1)
        )
        scene_flow_world = world_map_deform - world_map[:-1] # T-1, 3, H*W
        scene_flow_world = torch.cat(
            [
                scene_flow_world,
                torch.zeros(
                    1, 3, height * width, device=scene_flow_world.device
                ),
            ],
            dim=0,
        )
        scene_flow_world = scene_flow_world.permute(0, 2, 1).reshape(batch_size, height, width, 3) # T, H, W, 3
        return scene_flow_world

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        point_map = input_dict['point_map'].permute(0, 2, 3, 1)
        valid_mask = (input_dict['valid_mask'] > 0.5).float()
        camera_pose = input_dict['camera_pose']
        world_map = self.point_map_to_world(point_map, camera_pose)  # T, H, W, 4
        world_map = world_map * valid_mask.unsqueeze(-1)
        output_dict['point_map'] = world_map.permute(0, 3, 1, 2)

        if 'scene_flow' in input_dict and input_dict['scene_flow'] is not None:
            scene_flow = input_dict['scene_flow'].permute(0, 2, 3, 1)
            scene_flow_world = self.scene_flow_to_world(scene_flow, camera_pose, point_map, world_map)
            scene_flow_world = scene_flow_world * valid_mask.unsqueeze(-1)
            if "deform_mask" in input_dict and input_dict["deform_mask"] is not None:
                deform_mask = (input_dict['deform_mask'] > 0.5).float()
                scene_flow_world = scene_flow_world * deform_mask.unsqueeze(-1)
            output_dict['scene_flow'] = scene_flow_world.permute(0, 3, 1, 2)
            
        return output_dict

class ParamidPadding(nn.Module):
    def __init__(self, dk=2):
        super().__init__()
        self.dk = dk


    @torch.no_grad()
    def _build_pyramid(self, point_map, valid_mask):
        """
        Build a pyramid of premultiplied point_map * mask and mask itself.
        Args:
            point_map: (B, 3, H, W)
            valid_mask: (B, 1, H, W)
        Returns:
            List of tuples (premultiplied_point, mask) from small to large.
        """
        pyramid = []
        cur_p = point_map * valid_mask
        cur_m = valid_mask
        B, _, H, W = cur_m.shape

        while True:
            pyramid.append((cur_p, cur_m))
            if min(H, W) <= 1:
                break
            new_H, new_W = max(1, int(H / self.dk)), max(1, int(W / self.dk))
            cur_p = Fnn.interpolate(cur_p, size=(new_H, new_W), mode='area')
            cur_m = Fnn.interpolate(cur_m, size=(new_H, new_W), mode='area')
            H, W = new_H, new_W

        return pyramid[::-1]

    def pad(self, point_map, valid_mask):
        """
        Args:
            point_map: (B, 3, H, W)
            valid_mask: (B, H, W)
        Returns:
            padded_point_map: (B, H, W, 3)
        """
        pyramid = self._build_pyramid(point_map, valid_mask.unsqueeze(1).float())

        # top layer background mean (per batch)
        top_p, top_m = pyramid[0]
        num = top_p.sum(dim=(2, 3), keepdim=True)
        den = top_m.sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        fg = num / den  # (B, 3, 1, 1)

        # progressively upsample and blend
        for layer_p, layer_m in pyramid:
            _, _, H, W = layer_p.shape
            fg = Fnn.interpolate(fg, size=(H, W), mode='bilinear', align_corners=False)
            fg = layer_p + fg * (1.0 - layer_m)

        return fg

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        point_map = input_dict['point_map']
        valid_mask = input_dict['valid_mask'] > 0.5
        point_map = self.pad(point_map, valid_mask)
        output_dict['point_map'] = point_map

        # if "scene_flow" in input_dict and input_dict["scene_flow"] is not None:
        #     scene_flow = input_dict['scene_flow']
        #     scene_flow = self.pad(scene_flow, valid_mask)
        #     output_dict['scene_flow'] = scene_flow

        return output_dict


class DisparityNormalize(nn.Module):
    def __init__(self, is_normalized=False):
        super().__init__()
        self.is_normalized = is_normalized

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        disp = input_dict['disparity']
        if not self.is_normalized:
            disp = (disp - disp.min()) / (disp.max() - disp.min()+1e-4)
        disp = disp * 2.0 - 1.0  # -> [-1, 1]
        output_dict['disparity'] = disp
        return output_dict
    
class MaskNormalize(nn.Module):
    def __init__(self, neg=-1, pos=1):
        super().__init__()
        self.neg = neg
        self.pos = pos

    def forward(self, input_dict: Dict) -> Dict:
        output_dict = input_dict.copy()
        output_dict['valid_mask'] = torch.where(input_dict['valid_mask'] > 0.5, self.pos, self.neg)
        if 'deform_mask' in input_dict and input_dict['deform_mask'] is not None:
            output_dict['deform_mask'] = torch.where(input_dict['deform_mask'] > 0.5, self.pos, self.neg)
        return output_dict