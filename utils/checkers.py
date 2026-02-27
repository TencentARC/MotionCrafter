"""Utility helpers for NaN/Inf validation in tensors and arrays."""

import torch
import numpy as np

def check_isnan(tensor, tensor_name, *args):
    """Print diagnostics when a tensor/array contains NaN or Inf values."""
    if isinstance(tensor, torch.Tensor):
        # Check PyTorch tensor for NaN values
        if torch.isnan(tensor).any().item():
            print(f"{tensor_name} has Nan values")
            print(*args)
        # Check PyTorch tensor for Inf values
        elif torch.isinf(tensor).any().item():
            print(f"{tensor_name} has Inf values")
            print(*args)
    elif isinstance(tensor, np.ndarray):
        # Check NumPy array for NaN values
        if np.isnan(tensor).any():
            print(f"{tensor_name} has Nan values")
            print(*args)
        # Check NumPy array for Inf values
        elif np.isinf(tensor).any().item():
            print(f"{tensor_name} has Inf values")
            print(*args)
    else:
        raise NotImplementedError