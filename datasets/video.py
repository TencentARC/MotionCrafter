"""Video dataset module for loading and processing video data with geometric annotations.

This module provides the Video dataset class for training motion generation models.
It handles loading video frames, point maps, camera poses, scene flow, and other
geometric data from disk, with support for various data augmentation strategies.
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import random
import gin
import h5py
from typing import *

from utils.geo_utils import point_map_xy2intrinsic_map
from .video_transforms import *

@gin.configurable(denylist=[
    'sample_frame',
    'normalize_point_map', 
    'normalize_camera_pose',
    'pointmap_to_world', 
    'normalize_world_map',
    'normalize_disparity', 
    'normalize_valid_mask',
    'return_intrinsic_map', 
    'use_transform', 
    'resolution', ])
class Video(Dataset):
    """Video dataset class for loading and processing video frames with geometric data.
    
    This dataset class handles loading video frames from MP4 files and associated 
    geometric data (point maps, camera poses, scene flow, etc.) from HDF5 files. 
    It supports various augmentations and normalization options for training deep 
    learning models for motion and geometry estimation.
    
    The dataset can load:
    - RGB video frames from MP4 files
    - 3D point maps (dense depth + camera parameters)
    - Valid/invalid masks for point maps
    - Camera intrinsic and extrinsic parameters
    - Scene flow (3D motion between frames)
    - Pre-computed latent representations (optional)
    
    Data Augmentation:
    - Random resolution selection within specified ranges
    - Horizontal flipping with proper geometric transformation
    - Cover resize and center/random cropping
    - Color jittering for photometric augmentation
    - Frame sampling with configurable stride and window size
    
    Args:
        data_dir: Root directory containing video MP4 files and HDF5 data files.
        latent_dir: Optional directory containing pre-computed latent representations
                    from VAE encoding to speed up training.
        latent_only: If True, only loads latent representations without raw frames.
        sample_frame: If True, samples random frames; otherwise loads sequential clips.
        normalize_point_map: Whether to normalize point maps by mean depth.
        cuboid: Whether to use cuboid normalization for world coordinates.
        normalize_camera_pose: Whether to normalize camera poses.
        pointmap_to_world: Whether to convert point maps to world coordinates.
        normalize_world_map: Whether to normalize world coordinate maps.
        rescale: Whether to rescale point maps during normalization.
        reshift: Whether to recenter point maps during normalization.
        normalize_disparity: Whether to normalize disparity values.
        paramid_padding: Whether to use pyramid padding for augmentation.
        normalize_valid_mask: Whether to normalize validity masks.
        return_intrinsic_map: Whether to return camera intrinsic parameter maps.
        use_transform: Whether to apply data augmentation transforms.
        use_norm_data: Whether to use normalized data from disk if available.
        resolution: Target resolution as (height, width) tuple.
        resolution_range: Range of resolutions for random selection as [min, max].
        area_limit: Maximum pixel area constraint for resolution selection.
        resolution_choices: List of specific resolution tuples to randomly choose from.
        horizontal_flip: Probability of horizontal flip augmentation (0.0-1.0).
        resize: Resize strategy - 'cover' to cover target size or 'contain' to fit within.
        size_ratio_limit: Maximum ratio for random resize augmentation (>= 1.0).
        crop: Crop strategy - 'center' or 'random'.
        color_jittor: Whether to apply color jittering augmentation.
        use_frame_aug: Whether to apply frame-level augmentation.
        video_length_range: Range of video clip lengths as (min, max) frames.
        frame_stride_range: Range of frame sampling strides as (min, max).
        downsample_ratio: Downsampling ratio for video frames (optional).
        
    Attributes:
        sample_names: List of sample identifiers loaded from metadata file.
        latent_cache: Dictionary caching loaded latent representations.
        
    Example:
        >>> dataset = Video(
        ...     data_dir="data/train",
        ...     resolution=(320, 576),
        ...     horizontal_flip=0.5,
        ...     use_transform=True,
        ...     video_length_range=(14, 25)
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['frame', 'point_map', 'valid_mask', 'scene_flow', ...])
    """
    
    def __init__(
        self, 
        data_dir,
        latent_dir=None,
        latent_only=False,
        sample_frame=False,
        normalize_point_map=True,
        cuboid=True,
        normalize_camera_pose=True,
        pointmap_to_world=True,
        normalize_world_map=True,
        rescale=True,
        reshift=True,
        normalize_disparity=True,
        paramid_padding=True,
        normalize_valid_mask=True,
        return_intrinsic_map=False,
        use_transform=True,
        use_norm_data=True,
        resolution=(320, 640),
        resolution_range:List[int]=None,
        area_limit:int = None,
        resolution_choices:List[Tuple[int]]=None,
        horizontal_flip=False,
        resize='cover',
        size_ratio_limit=1.2,
        crop='center',
        color_jittor=False,
        use_frame_aug=True,
        video_length_range=(1, 25),
        frame_stride_range=(1, 8),
        downsample_ratio=None
    ):
        super().__init__()
        
        # Store configuration parameters
        self.data_dir = data_dir
        self.latent_dir = latent_dir
        self.latent_only = latent_only
        self.sample_frame = sample_frame
        self.normalize_point_map = normalize_point_map
        self.normalize_camera_pose = normalize_camera_pose
        self.pointmap_to_world = pointmap_to_world
        self.normalize_world_map = normalize_world_map
        self.normalize_disparity = normalize_disparity
        self.normalize_valid_mask = normalize_valid_mask
        self.return_intrinsic_map = return_intrinsic_map

        self.resolution = resolution
        self.resolution_range = resolution_range
        self.area_limit = area_limit
        self.resolution_choices = resolution_choices

        # Load metadata about samples
        if os.path.exists(os.path.join(data_dir, 'meta_infos.txt')):
            meta_file_path = os.path.join(data_dir, 'meta_infos.txt')
        else:
            meta_file_path = os.path.join(data_dir, 'filename_list.txt')
        assert os.path.exists(meta_file_path), meta_file_path

        self.samples = []        
        with open(meta_file_path, "r") as f:
            for line in f.readlines():
                video_path, data_path = line.split()[:2]
                if use_norm_data:
                    # Use normalized data files if available
                    video_path = video_path.replace('_rgb.mp4', f'_rgb_{resolution[0]}_{resolution[1]}.mp4')
                    data_path = data_path.replace('_data.hdf5', f'_normed_data_{resolution[0]}_{resolution[1]}.hdf5')
                self.samples.append(dict(
                    video_path=video_path,
                    data_path=data_path
                ))

        if downsample_ratio is not None and downsample_ratio < 1.0:
            # Downsample the dataset by selecting every nth sample
            step = int(1.0 / downsample_ratio)
            ori_len = len(self.samples)
            self.samples = [self.samples[i] for i in range(0, ori_len, step)]
            self.samples = self.samples[:int(ori_len*downsample_ratio)]

        if use_transform:
            # Build transformation pipeline with various augmentations and normalizations
            assert not self.latent_only
            trans = []
            if horizontal_flip:
                trans.append(RandomHorizontalFlip())
            if resize == "cover":
                trans.append(CoverResize(antialias=True))
            elif resize == "random":
                trans.append(RandomResize(size_ratio_limit=size_ratio_limit, antialias=True))
            else:
                assert resize == "none"
            if crop == "random":
                trans.append(RandomCrop())
            elif crop == "center":
                trans.append(CenterCrop())
            else:
                assert crop == "none"                
            if color_jittor:
                trans.append(ColorJitter(brightness=[0.2, 1], contrast=0., saturation=0., hue=0.))
            if normalize_point_map:
                trans.append(PointMapNormalize(use_norm_data, cuboid=cuboid))
            if normalize_camera_pose:
                trans.append(CameraPoseNormalize())
            if pointmap_to_world:
                trans.append(PointMaptoWorld())
            if normalize_world_map:
                trans.append(WorldMapNormalize(rescale, reshift))
            # if normalize_disparity:
            #     trans.append(DisparityNormalize(use_norm_data))
            if paramid_padding:
                trans.append(ParamidPadding())
            if normalize_valid_mask:
                trans.append(MaskNormalize(-1, 1))
            else:
                trans.append(MaskNormalize(0, 1))
            
            self.transform = Compose(trans)
        else:
            self.transform = None

        self.depth_eps = 1e-5
    
        self.use_frame_aug = use_frame_aug
        # Store possible video lengths and frame strides for sampling
        self.video_length_choice = list(range(video_length_range[0], video_length_range[1] + 1))
        if frame_stride_range != -1:
            self.frame_stride_choice = list(
                range(frame_stride_range[0], frame_stride_range[1] + 1)
            )
        else:
            # -1 means random frame sampling without fixed stride
            self.frame_stride_choice = -1

    def __len__(self):
        # Return total number of samples in the dataset
        return len(self.samples)

    def _get_output_resolution(self, block_size = 64):
        # Ensure height/width can be divided by block_size (e.g., 64 for SVD model)
        if self.resolution is not None:
            resolution = self.resolution
        elif self.resolution_range is not None:
            height = random.randint(
                self.resolution_range[0], 
                self.resolution_range[1]+block_size-1,
            )
            height = height // block_size * block_size
            if self.area_limit:
                max_width = int(self.area_limit // height)
            else:
                max_width = 1e9
            
            width = random.randint(
                self.resolution_range[2], 
                min(max_width, self.resolution_range[3])+block_size-1
            )
            width = width // block_size * block_size
            resolution = (height, width)     
        elif self.resolution_choices is not None:
            rand_idx = random.randint(
                0, 
                len(self.resolution_choices)-1,
            )
            resolution = self.resolution_choices[rand_idx]
        else:
            resolution = None
        return resolution

    def __getitem__(self, index):
        # Load and process a single sample by index
        sample = self.samples[index]
        if self.latent_dir:
            latent_path = osp.join(self.latent_dir, '.'.join(sample["data_path"].split('.')[:-1])+'.npz')

        if self.latent_only:
            sample = self._load_latent_npz(latent_path)
            sample = self._to_float_tensor(sample)
            return sample
        
        video_path = sample['video_path']
        data_path = sample['data_path']
        sample = self._get_output_numpy(sample)
        sample = self._to_float_tensor(sample)

        frame = (sample.pop('frame') / 255 - 0.5) * 2  # Convert [0, 255] to [-1, 1]
        # disp = sample.pop('disparity')
        valid_mask = sample.pop('valid_mask')
        point_map = sample.pop('point_map')
        camera_pose = sample.pop('camera_pose')
        scene_flow = sample.pop('scene_flow', None)
        deform_mask = sample.pop('deform_mask', None)
        
        data = dict(
            video_path=video_path,
            data_path=data_path,
            index=index,
            frame=frame.permute(0, 3, 1, 2),
            # disparity=disp,
            valid_mask=valid_mask,
            point_map=point_map.permute(0, 3, 1, 2),
            camera_pose=camera_pose,
        )

        if scene_flow is not None:
            data['scene_flow'] = scene_flow.permute(0, 3, 1, 2)
        if deform_mask is not None:
            data['deform_mask'] = deform_mask

        if self.transform:
            data['resolution'] = self._get_output_resolution()
            data = self.transform(data)
            data.pop('resolution')

        if self.return_intrinsic_map:
            if self.normalize_point_map:
                point_data = data['point_map'].permute(0, 2, 3, 1)[..., :2]
                intrinsic_map = point_map_xy2intrinsic_map(point_data)    
            else:
                point_data = data['point_map'].permute(0, 2, 3, 1)
                intrinsic_map = point_map_xy2intrinsic_map(
                    point_data[..., :2] / point_data[..., 2:3]
                )
            # intrinsic_map = intrinsic_map.permute(0, 3, 1, 2) * (data['valid_mask'] > 0.5).unsqueeze(1)
            intrinsic_map = intrinsic_map.permute(0, 3, 1, 2)
            data['intrinsic_map'] = intrinsic_map    

        if self.latent_dir:
            data['latent_path'] = latent_path

        if self.sample_frame:
            # Sample a single frame from the sequence
            sample_idx = random.randint(0, len(data['frame']) - 1)
            data['frame'] = data['frame'][sample_idx:sample_idx+1]
            # data['disparity'] = data['disparity'][sample_idx:sample_idx+1]
            data['valid_mask'] = data['valid_mask'][sample_idx:sample_idx+1]
            data['point_map'] = data['point_map'][sample_idx:sample_idx+1]
            data['camera_pose'] = data['camera_pose'][sample_idx:sample_idx+1]
            if scene_flow is not None:
                data['scene_flow'] = data['scene_flow'][sample_idx:sample_idx+1]
            if deform_mask is not None:
                data['deform_mask'] = data['deform_mask'][sample_idx:sample_idx+1]

        return data

    def _to_float_tensor(self, sample):
        # Convert numpy arrays in sample dict to float tensors
        output_tensor = dict()
        for k, v in sample.items():
            if v is None:
                output_tensor[k] = None
            else:
                output_tensor[k] = torch.tensor(v).float()
        return output_tensor

    def _get_frame_indices(self, frame_num):
        # Select frame indices based on video length and frame stride settings
        video_length = random.choice(self.video_length_choice)

        if self.frame_stride_choice == -1:
            # Random sampling without fixed stride
            if video_length > frame_num:
                return [random.choice(list(range(frame_num))) for i in range(video_length)]
            else:
                return random.sample(list(range(frame_num)), video_length)
        # Fixed stride sampling
        frame_stride = random.choice(self.frame_stride_choice)
        frame_stride = min(frame_stride, frame_num // video_length)
        frame_stride = max(frame_stride, 1)
        required_frame_num = frame_stride * (video_length - 1) + 1
        if required_frame_num <= frame_num:
            random_range = frame_num - required_frame_num
            start_idx = (
                random.randint(0, random_range) if random_range > 0 else 0
            )
            frame_indices = [
                start_idx + frame_stride * i for i in range(video_length)
            ]
        else:
            frame_indices = list(range(frame_num))
            if len(frame_indices) >= 3:
                dummy_indices = frame_indices[1:-1]
            else:
                dummy_indices = frame_indices
            while len(frame_indices) < video_length:
                dummy_indices = dummy_indices[::-1]
                frame_indices = frame_indices + dummy_indices
            frame_indices = frame_indices[:video_length]
        return frame_indices

    def _load_latent_npz(self, latent_path):
        # Load pre-computed latent representations from NPZ file
        sample = np.load(latent_path)
        data = dict()
        if self.use_frame_aug:
            # Select frames based on augmentation settings
            frame_indices = self._get_frame_indices(sample["conditional_latents"].shape[0])        
            for k in sample:
                if k not in ['noise_aug_strength']:
                    data[k] = sample[k][frame_indices]
                else:
                    data[k] = sample[k]
        else:
            for k in sample:
                data[k] = sample[k]
        return data

    def _get_output_numpy(self, sample):
        # Load video frames and geometric data from disk
        video_reader = VideoReader(osp.join(self.data_dir, sample["video_path"]), ctx=cpu(0))
        if self.use_frame_aug:
            # Load selected frames and corresponding geometric data
            frame_indices = self._get_frame_indices(len(video_reader))
            frame = video_reader.get_batch(frame_indices)
            valid_masks, point_maps, scene_flows, camera_poses, deform_masks = [], [], [], [], []
            with h5py.File(osp.join(self.data_dir, sample["data_path"]), "r") as file:
                for i in frame_indices:
                    # Load data for each frame
                    # disps.append(file['disparity'][i,:])
                    valid_masks.append(file['valid_mask'][i,:])
                    point_maps.append(file['point_map'][i,:])
                    camera_poses.append(file['camera_pose'][i,:])
                    if 'scene_flow' in file:
                        # Last frame does not have scene flow (no next frame)
                        if i != frame_indices[-1]:
                            scene_flows.append(file['scene_flow'][i,:])
                        else:
                            scene_flows.append(np.zeros_like(file['point_map'][i,:]))
                    if 'deform_mask' in file:
                        deform_masks.append(file['deform_mask'][i,:])

            # disps = np.stack(disps)
            valid_masks = np.stack(valid_masks)
            point_maps = np.stack(point_maps)
            camera_poses = np.stack(camera_poses)
            scene_flows = np.stack(scene_flows) if len(scene_flows) > 0 else None
            deform_masks = np.stack(deform_masks) if len(deform_masks) > 0 else None
        else:
            frame = video_reader.get_batch(
                list(range(len(video_reader)))
            )
            with h5py.File(osp.join(self.data_dir, sample["data_path"]), "r") as file:
                # disps = file['disparity'][:]
                valid_masks = file['valid_mask'][:]
                point_maps = file['point_map'][:]
                try:
                    camera_poses = file['camera_pose'][:]
                except KeyError:
                    print(f"Warning: camera_pose not found in {sample['data_path']}, using identity matrix.")
                    camera_poses = np.tile(np.eye(4)[None, :, :], (point_maps.shape[0], 1, 1))
                try:
                    scene_flows = file['scene_flow'][:]
                except KeyError:
                    scene_flows = None
                try:
                    deform_masks = file['deform_mask'][:]
                except KeyError:
                    deform_masks = None

        return dict(
            frame=frame.asnumpy(),
            point_map=point_maps,  # T,H,W,3
            valid_mask=valid_masks,  # T,H,W
            # disparity=disps,  # T,H,W
            camera_pose=camera_poses,  # T,4,4
            scene_flow=scene_flows,  # T,H,W,3 (optional)
            deform_mask=deform_masks,  # T,H,W (optional)
        )