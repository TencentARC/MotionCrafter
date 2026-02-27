"""Dataset factory and composite dataset definitions for MotionCrafter training."""

from dataclasses import dataclass
from typing import Tuple, List
import gin
from torch.utils.data import Dataset
import numpy as np
import torch
import os

from .video import Video

@gin.configurable(denylist=[
    'normalize_point_map',
    'normalize_camera_pose',
    'pointmap_to_world', 
    'normalize_world_map',
    'normalize_disparity', 
    'normalize_valid_mask', 
    'return_intrinsic_map',
    'use_transform', 
    'resolution',])
class CompositeVideoDataset(Dataset):
    """
    A composite dataset class that manages multiple video datasets.
    
    This class combines multiple Video datasets and provides unified access
    through a single dataset interface. It supports various augmentation options
    including normalization, resizing, cropping, and color jittering.
    """
    def __init__(
        self, 
        data_dirs,  # List of directories containing video data
        latent_dirs=None,  # Optional list of directories for latent representations
        latent_only=False,  # Whether to use only latent representations
        sample_frame=False,  # Whether to sample frames during loading
        normalize_point_map=True,  # Normalize point cloud/depth maps
        normalize_camera_pose=True,  # Normalize camera pose parameters
        pointmap_to_world=True,  # Convert point maps to world coordinates
        normalize_world_map=True,  # Normalize world coordinate maps
        rescale=True,  # Whether to rescale video data
        reshift=True,  # Whether to re-center/shift data
        paramid_padding=True,  # paramid padding if needed
        normalize_disparity=True,  # Normalize disparity/depth values
        normalize_valid_mask=True,  # Normalize validity masks
        return_intrinsic_map=False,  # Whether to return camera intrinsics
        use_transform=True,  # Apply transformations
        resolution:Tuple[int]=(320, 640),  # Target resolution (height, width)
        resolution_range:List[int]=None,  # Range of allowed resolutions
        area_limit:int = None,  # Limit on valid area percentage
        resolution_choices:List[Tuple[int]]=None,  # List of possible resolutions
        horizontal_flip=False,  # Apply horizontal flipping augmentation
        resize='cover',  # Resizing strategy: 'cover' or 'contain'
        size_ratio_limit=1.2,  # Maximum size ratio for augmentation
        crop='center',  # Cropping strategy: 'center', 'random', etc.
        color_jittor=False,  # Apply color jittering augmentation
        use_frame_aug=True,  # Use frame-level augmentations
        video_length_range=(1, 25),  # Range of video sequence lengths
        frame_stride_range=(1, 8),  # Range of frame sampling strides
        downsample_ratios=None,  # Downsampling ratios for each dataset
        return_info=True,  # Whether to return dataset and index information
        filtered_low_quality=False,  # Filter out low quality samples
        debug=False  # Debug mode for dataset inspection
    ):
        # Initialize dataset configuration and state variables
        self.debug = debug
        self.normalize_valid_mask = normalize_valid_mask
        self.datasets = []
        self.dataset_list = []
        self.return_info = return_info
        self.filtered_low_quality = filtered_low_quality
        
        # Load and initialize individual Video datasets
        for idx, data_dir in enumerate(data_dirs): 
            # Extract dataset name from directory path
            self.dataset_list.append(os.path.basename(data_dir[:-1] if data_dir[-1] == '/' else data_dir))
            # Create a Video dataset instance for each data directory
            self.datasets.append(
                Video(
                    data_dir=data_dir,
                    latent_dir=latent_dirs[idx] if latent_dirs is not None else None,
                    latent_only=latent_only,
                    sample_frame=sample_frame,
                    normalize_point_map=normalize_point_map,
                    normalize_camera_pose=normalize_camera_pose,
                    pointmap_to_world=pointmap_to_world,
                    normalize_world_map=normalize_world_map,
                    rescale=rescale,
                    reshift=reshift,
                    paramid_padding=paramid_padding,
                    normalize_disparity=normalize_disparity,
                    normalize_valid_mask=normalize_valid_mask,
                    return_intrinsic_map=return_intrinsic_map,
                    use_transform=use_transform,
                    resolution=resolution,
                    resolution_range=resolution_range,
                    area_limit=area_limit,
                    resolution_choices=resolution_choices,
                    horizontal_flip=horizontal_flip,
                    resize=resize,
                    size_ratio_limit=size_ratio_limit,
                    crop=crop,
                    color_jittor=color_jittor,
                    use_frame_aug=use_frame_aug,
                    video_length_range=video_length_range,
                    frame_stride_range=frame_stride_range,
                    downsample_ratio=downsample_ratios[idx] if downsample_ratios is not None else None
                )
            )

        # Calculate dataset statistics and index boundaries
        self.len_datasets = [len(dataset) for dataset in self.datasets]
        self.st_indices = []
        self.ed_indices = []
        last_idx = 0
        for l in self.len_datasets:
            self.st_indices.append(last_idx)
            self.ed_indices.append(last_idx+l)
            last_idx += l

    def __len__(self):
        return sum(self.len_datasets)

    def __getitem__(self, index, max_retries=3):
        # Fetch an item by composite index and handle errors with retries
        if max_retries <= 0:
            raise RuntimeError("Max retries exceeded while fetching item from composite video dataset.")
        
        # Find which dataset this index belongs to
        for i in range(len(self.datasets)):
            if self.st_indices[i] <= index and index < self.ed_indices[i]:
                # In debug mode, return only dataset name and local index
                if self.debug:
                    return dict(
                        dataset=self.dataset_list[i],
                        index=index - self.st_indices[i],
                    )
                try:
                    # Fetch data from the corresponding dataset using local index
                    data = self.datasets[i][index - self.st_indices[i]]
                    # Validate tensor values for NaN and Inf
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            assert not v.isnan().any(), f'Key {k} has nan value'
                            assert not v.isinf().any(), f'Key {k} has inf value'
                    # Filter low-quality samples if specified
                    if self.filtered_low_quality and 'valid_mask' in data:
                        threshold = 0 if self.normalize_valid_mask else 0.5
                        assert (data['valid_mask'] > threshold).sum() >= 0.1 * np.prod(data['valid_mask'].shape), \
                            f'Too many invalid pixels in sample {index} from dataset {self.dataset_list[i]}'
                    # Add dataset information if requested
                    if self.return_info:
                        data['info'] = f"{self.dataset_list[i]}_{index - self.st_indices[i]}"
                    return data
                except Exception as e:
                    # On error, print error message and retry with a random sample
                    print(e)
                    print(f'Error during fetch {index - self.st_indices[i]} from dataset {self.dataset_list[i]}')
                    return self.__getitem__(np.random.randint(0, len(self)), max_retries=max_retries-1)
        raise NotImplementedError


@gin.configurable()
@dataclass
class DatasetConfig:
    """
    Configuration dataclass for dataset initialization.
    
    This class encapsulates all configuration parameters for creating
    a CompositeVideoDataset instance. It supports Gin configuration
    for flexible hyperparameter management.
    """
    __dataset__: str = "CompositeVideoDataset"  # Dataset class name to instantiate
    
    normalize_point_map: bool = False  # Normalize 3D point coordinates
    normalize_camera_pose: bool = True  # Normalize camera extrinsics
    pointmap_to_world: bool = True  # Convert point maps to world frame
    normalize_world_map: bool = True  # Normalize world coordinate maps
    rescale: bool = True  # Apply rescaling to data
    reshift: bool = True  # Apply re-centering to data
    paramid_padding: bool = True  # Paramid padding if needed
    normalize_disparity: bool = True  # Normalize depth/disparity values
    normalize_valid_mask: bool = True  # Normalize validity mask values
    return_intrinsic_map: bool = False  # Include camera intrinsic matrices
    use_transform: bool = True  # Apply transforms
    resolution: Tuple[int] = (320, 640)  # Output resolution (H, W)
    batch_sampler: str = 'random'  # Sampling strategy for batches

    def build(self):
        """
        Build and return a dataset instance from this configuration.
        
        Returns:
            CompositeVideoDataset: An instantiated dataset with the configured parameters
        """
        return globals()[self.__dataset__](
            normalize_point_map=self.normalize_point_map,
            normalize_camera_pose=self.normalize_camera_pose,
            pointmap_to_world=self.pointmap_to_world,
            normalize_world_map=self.normalize_world_map,
            rescale=self.rescale,
            reshift=self.reshift,
            paramid_padding=self.paramid_padding,
            normalize_disparity=self.normalize_disparity,
            normalize_valid_mask=self.normalize_valid_mask,
            return_intrinsic_map=self.return_intrinsic_map,
            use_transform=self.use_transform,
            resolution=self.resolution
        )
