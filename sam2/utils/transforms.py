# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        
        
        # Only normalize is torchscript-ed now
        self.transforms = torch.jit.script(
            nn.Sequential(
                Normalize(self.mean, self.std),
            )
        )

        self.params_set = False

    def _set_hw_params(self, image, image_size):
        self.resized_image_size, self.orig_hw = get_resize_longest_side(image, image_size)
        # Custom resize and pad function that can't be torchscript-ed
        self.resize_pad = ResizeLongestSide(image_size, self.resized_image_size)
        padding = self.resize_pad.get_padding()
        self.params_set = True

        return self.resized_image_size, padding

    def __call__(self, x):
        if not self.params_set:
            raise ValueError("Parameters not set. Call _set_hw_params first.")
        x = self.to_tensor(x)
        x = self.resize_pad(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert self.orig_hw is not None
            h, w = self.orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        h, w = masks.shape[-2:]
        masks = F.interpolate(masks, (h*4, w*4), mode="bilinear", align_corners=False)
        
        # Get the dimensions from resized_image_size
        new_h, new_w = self.resized_image_size
        
        # Calculate padding offsets to crop from the center
        h_total = h * 4
        w_total = w * 4
        h_offset = (h_total - new_h) // 2
        w_offset = (w_total - new_w) // 2
        
        # Remove padding by cropping from the center
        masks = masks[..., h_offset:h_offset + new_h, w_offset:w_offset + new_w]
        
        # Finally resize to original image dimensions
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

def get_resize_longest_side(x, target_length):
    if isinstance(x, np.ndarray):
        h, w = x.shape[:2]
    elif isinstance(x, torch.Tensor):
        h, w = x.shape[-2:]
    elif isinstance(x, Image.Image):
        w, h = x.size  
    else:
        raise ValueError(f"Unsupported input type: {type(x)}")
    
    scale = target_length / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return (new_h, new_w), (h, w)

class ResizeLongestSide(nn.Module):
    def __init__(self, target_length, resized_image_size):
        super().__init__()
        self.target_length = target_length
        self.new_h, self.new_w = resized_image_size

    def get_padding(self):
        
        # Pad
        pad_h = self.target_length - self.new_h
        pad_w = self.target_length - self.new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        # Resize
        x = F.interpolate(x.unsqueeze(0), size=(self.new_h, self.new_w), mode='bilinear', align_corners=False).squeeze(0)

        pad_left, pad_right, pad_top, pad_bottom = self.get_padding()
        
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
