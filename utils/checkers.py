# Tensor Validation Utilities
# Provides functions to check for NaN and Inf values in tensors

import torch
import numpy as np

def check_isnan(tensor, tensor_name, *args):
    # Check if tensor contains NaN or Inf values
    # Args:
    #   tensor: Tensor to check (torch.Tensor or np.ndarray)
    #   tensor_name: Name of tensor for error reporting
    #   *args: Additional arguments to print if NaN/Inf detected
    # Raises:
    #   NotImplementedError: If tensor type is not supported
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