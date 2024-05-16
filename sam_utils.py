import numpy as np
import torch
from typing import Optional, Type

def combine_masks_to_single_layer(masks: np.ndarray | torch.Tensor, single_id: bool = False, output_dtype: Optional[Type] = None,
                                  max_objects_allowed: int = 255) -> np.ndarray | torch.Tensor:
    """
    Combine multiple object masks into a single layer. Converts array from [num_objects, H, W] to [H, W].

    Args:
    masks (np.ndarray): A boolean numpy array of shape [num_objects, H, W] representing object masks.
    single_id (bool): If True, assign all objects the same ID (1). If False, assign a unique ID to each object.
    output_dtype: Data type of the output mask. Default is uint8.
    max_objects_allowed (int): maximum objects allowed in mask

    Returns:
    np.ndarray: An array of shape [H, W] where each value is either 0 (background) or a non-zero integer representing a distinct object.
    """

    # If the input is already a single layer mask, return it as is
    if len(masks.shape) < 3:
        return masks
    
    num_objects, H, W = masks.shape
    is_numpy = isinstance(masks, np.ndarray)

    if output_dtype is None:
        output_dtype = np.uint8 if is_numpy else torch.uint8

    if is_numpy:
        output_mask = np.zeros((H, W), dtype=output_dtype)
    else:
        output_mask = torch.zeros((H, W), dtype=output_dtype)

    # Ensure the number of objects does not exceed the maximum value of the specified dtype
    # if np.issubdtype(output_dtype, np.floating) or (is_numpy == False and torch.is_floating_point(output_dtype)):
    #     max_objects_allowed = np.finfo(output_dtype).max if is_numpy else torch.finfo(output_dtype).max
    # else:
    #     max_objects_allowed = np.iinfo(output_dtype).max if is_numpy else torch.iinfo(output_dtype).max

    if num_objects > max_objects_allowed and not single_id:
        raise ValueError(f"Number of objects exceeds the maximum allowed ({max_objects_allowed})")


    for i in range(num_objects):
        object_id = 1 if single_id else (i + 1)
        # Assign a unique number to each object if not single_id
        if is_numpy:
            output_mask[masks[i] == 1] = object_id # 1 indicates the object
        else:
            output_mask[masks[i].bool()] = object_id

    return output_mask