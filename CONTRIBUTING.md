# Contributing to MotionCrafter

Thank you for your interest in contributing to MotionCrafter! This document provides guidelines for code documentation and style to maintain consistency across the project.

## Code Documentation Guidelines

### 1. Module-Level Documentation

Every Python file should start with a module-level docstring describing its purpose:

```python
"""Brief one-line description of the module.

Longer description providing more details about the module's functionality,
design decisions, and usage patterns. This can span multiple paragraphs.

Key features:
- Feature 1
- Feature 2

Example:
    >>> from module import ClassName
    >>> obj = ClassName()
"""
```

### 2. Class Documentation

Use Google-style docstrings for classes:

```python
class MyClass:
    """Brief description of the class.
    
    Longer description explaining the class's purpose, main responsibilities,
    and how it fits into the larger system.
    
    Attributes:
        attr1: Description of attribute 1.
        attr2: Description of attribute 2.
        
    Example:
        >>> obj = MyClass(param1="value")
        >>> result = obj.method()
    """
```

### 3. Function Documentation

Use comprehensive docstrings with Args, Returns, and Raises sections:

```python
def function_name(
    arg1: str,
    arg2: int = 10,
    arg3: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Brief description of what the function does.
    
    Longer description providing implementation details, algorithm explanation,
    or usage notes if needed.
    
    Args:
        arg1: Description of first argument. Explain its purpose and
              expected format.
        arg2: Description of second argument with default value.
        arg3: Optional argument description. Explain None behavior.
        
    Returns:
        Description of return value. For complex types, explain the
        structure, e.g., "Dictionary containing 'key1' (int) and 
        'key2' (str)."
        
    Raises:
        ValueError: When arg1 is invalid.
        RuntimeError: When operation fails.
        
    Example:
        >>> result = function_name("test", arg2=20)
        >>> print(result['key1'])
        42
        
    Note:
        Additional notes about edge cases, performance considerations,
        or important implementation details.
    """
```

### 4. Type Annotations

Always use type hints for function signatures:

```python
from typing import Optional, List, Dict, Union, Tuple, Any

def process_data(
    data: torch.Tensor,
    config: Dict[str, Any],
    device: Optional[str] = None
) -> Tuple[torch.Tensor, bool]:
    """Process input data with configuration."""
    pass
```

### 5. Inline Comments

Use inline comments to explain complex logic, but avoid obvious comments:

**Good:**
```python
# Apply EDM-style log-normal noise schedule for better sampling
sigmas = rand_log_normal(shape=[B,], loc=p_mean, scale=p_std)

# Normalize by mean valid depth to ensure scale consistency
norm_factor = (point_map[..., 2] * valid_mask.float()).mean()
```

**Bad:**
```python
# Set x to 5
x = 5

# Loop through items
for item in items:
    process(item)
```

### 6. TODOs and FIXMEs

Format action items clearly:

```python
# TODO(username): Add support for dynamic batch sizes
# FIXME: Memory leak when processing > 100 frames
# NOTE: This workaround addresses issue #123
```

## Code Style Guidelines

### 1. Naming Conventions

- **Classes**: `PascalCase` (e.g., `MotionCrafterPipeline`)
- **Functions/Methods**: `snake_case` (e.g., `encode_video`, `compute_loss`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_FRAMES`, `DEFAULT_HEIGHT`)
- **Private members**: `_leading_underscore` (e.g., `_internal_method`)

### 2. Import Organization

Organize imports in the following order:

```python
"""Module docstring."""

# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, List

# 2. Third-party imports
import torch
import numpy as np
from diffusers import AutoencoderKL

# 3. Local application imports
from motioncrafter import BasePipeline
from utils.geo_utils import normalize_point_map
```

### 3. Line Length and Formatting

- Maximum line length: 100 characters (prefer 88 for compatibility with Black formatter)
- Use 4 spaces for indentation (never tabs)
- Two blank lines between top-level definitions
- One blank line between method definitions

### 4. String Formatting

Prefer f-strings for string formatting:

```python
# Good
message = f"Processing frame {frame_idx} of {total_frames}"

# Avoid
message = "Processing frame {} of {}".format(frame_idx, total_frames)
message = "Processing frame " + str(frame_idx) + " of " + str(total_frames)
```

## Documentation Best Practices

### 1. Be Specific and Accurate

```python
# Good
"""Compute weighted MSE loss between predicted and ground truth point maps.
Valid points are weighted by their inverse distance to avoid bias toward 
far-away points."""

# Too vague
"""Compute loss."""
```

### 2. Document Edge Cases

```python
def resize_tensor(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize tensor to target size.
    
    Args:
        tensor: Input tensor of shape (B, C, H, W).
        size: Target size as (height, width).
        
    Returns:
        Resized tensor of shape (B, C, height, width).
        
    Note:
        If size matches input dimensions, returns input unchanged.
        For downsampling > 2x, applies antialiasing automatically.
    """
```

### 3. Provide Examples

Include usage examples for complex functions:

```python
def encode_vae_video(
    video: torch.Tensor,
    chunk_size: int = 14
) -> torch.Tensor:
    """Encode video frames to VAE latent space.
    
    Args:
        video: Video tensor of shape (T, C, H, W) in range [-1, 1].
        chunk_size: Number of frames to encode at once to manage memory.
        
    Returns:
        Latent tensor of shape (T, latent_C, H//8, W//8).
        
    Example:
        >>> video = torch.randn(25, 3, 256, 256)
        >>> latents = encode_vae_video(video, chunk_size=8)
        >>> print(latents.shape)
        torch.Size([25, 4, 32, 32])
    """
```

### 4. Maintain Consistency

Use consistent terminology throughout:
- "point map" not "depth map" or "3D points" (choose one)
- "scene flow" not "optical flow" or "motion field"
- "validity mask" not "valid mask" or "mask"

## Pre-commit Checklist

Before submitting code:

- [ ] All public functions have docstrings
- [ ] All docstrings include Args/Returns sections
- [ ] Type hints are present for all function arguments
- [ ] Complex logic has explanatory comments
- [ ] Examples are provided for non-trivial functions
- [ ] No commented-out code (use git history instead)
- [ ] TODO/FIXME items include author and description

## Tools

We recommend using the following tools:

- **Black**: Code formatter (line length 88)
- **isort**: Import sorting
- **mypy**: Static type checking
- **pylint**: Code linting

Example configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

## Questions?

If you have questions about documentation standards or need clarification, please:
1. Check existing code for examples
2. Refer to [PEP 257](https://peps.python.org/pep-0257/) for docstring conventions
3. See [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for comprehensive guidelines
4. Open an issue for discussion

Thank you for helping maintain high code quality in MotionCrafter!
