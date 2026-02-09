"""Image saving utilities for tensors and NumPy arrays."""

import torch
from PIL import Image
import numpy as np


def save_image_tensor(filename, image_tensor):
    """
    Save a CHW float tensor in [-1, 1] to an image file.

    Args:
        filename: Output image path.
        image_tensor: Tensor of shape (3, H, W) with values in [-1, 1].
    """
    # Convert to HWC format in [0, 1] range, then to uint8
    image = image_tensor.permute(1, 2, 0) / 2 + 0.5
    image = (torch.clamp(image, 0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(image).save(filename)


def save_image_numpy(filename, image_numpy):
    """
    Save an HWC NumPy array in [0, 1] to an image file.

    Args:
        filename: Output image path.
        image_numpy: Array of shape (H, W, 3) with values in [0, 1].
    """
    # Clip to valid range and convert to uint8
    image = (np.clip(image_numpy, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(filename)