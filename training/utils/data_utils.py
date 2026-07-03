# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    heatmaps: torch.FloatTensor
    metadata: BatchedVideoMetaData
    bkgd_masks: torch.BoolTensor
    dict_key: str
    cell_divides: torch.IntTensor
    cell_tracks_mask: torch.BoolTensor
    daughter_ids: torch.IntTensor
    no_inputs: torch.BoolTensor
    target_obj_mask: torch.BoolTensor
    is_real: torch.BoolTensor  # True for real objects, False for padded objects
    is_real_masks: torch.BoolTensor  # True for real masks, False for padded masks
    centroids: torch.FloatTensor  # [T, max_objects_masks, 2] (x, y) per mask, same order as masks

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return len(self.is_real)

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        Now handles tensor format [T, max_objects, 2] and filters out padded entries.
        """

        flat_idx = []

        for i in range(self.obj_to_frame_idx.shape[0]):
            # Filter out padded entries using is_real
            batch = self.obj_to_frame_idx[i][self.is_real[i]]
            frame_idx = batch[:,0]
            video_idx = batch[:,1]
            flat_idx.append(video_idx * self.num_frames + frame_idx)

        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask
    entering: Optional[bool] = None
    parent_id: Optional[int] = None
    daughter_ids: Optional[torch.Tensor] = None
    is_two_dau: Optional[bool] = None
    is_one_dau: Optional[bool] = None
    is_in_next_object_ids_list: Optional[bool] = None

@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    object_ids: List[int]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]
    man_track: torch.IntTensor
    clip_zoom_scale: float = 1.0

def pad_and_stack(tensor_list, max_objects, pad_value=0):
    """
    Pad tensors in list to same size (max_objects) and stack them along time dimension.
    
    Args:
        tensor_list: List of tensors, each with shape [num_objects, ...]
        max_objects: Maximum number of objects to pad to
        pad_value: Value to use for padding (default: 0)
    
    Returns:
        Stacked tensor with shape [T, max_objects, ...] where T is len(tensor_list)
    """
    padded = []
    for t in tensor_list:
        if t.shape[0] < max_objects:
            pad_size = max_objects - t.shape[0]
            device = t.device if hasattr(t, 'device') else torch.device('cpu')
            # Create padding tensor matching the shape
            if len(t.shape) == 1:
                padding = torch.full((pad_size,), pad_value, dtype=t.dtype, device=device)
            elif len(t.shape) == 2:
                padding = torch.full((pad_size, t.shape[1]), pad_value, dtype=t.dtype, device=device)
            elif len(t.shape) == 3:
                padding = torch.full((pad_size, t.shape[1], t.shape[2]), pad_value, dtype=t.dtype, device=device)
            else:
                padding = torch.full((pad_size,) + t.shape[1:], pad_value, dtype=t.dtype, device=device)
            t = torch.cat([t, padding], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)  # Shape: [T, max_objects, ...]

def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T,B,_,H,W = img_batch.shape
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    bkgd_masks = torch.zeros(T,B,H,W, dtype=torch.bool)

    step_t_cell_divides = [[] for _ in range(T)]
    step_t_cell_tracks_mask = [[] for _ in range(T)]
    step_t_target_obj_mask = [[] for _ in range(T)]
    step_t_daughter_ids = [[] for _ in range(T)]
    step_t_no_inputs = []
    step_t_centroids = [[] for _ in range(T)]

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            dividing_masks = {}
            dividing_centroids = {}
            for obj in objects:
                if obj.object_id == -1000:
                    bkgd_masks[t,video_idx] += obj.segment.to(torch.bool)
                    continue

                centroid = get_centroids_from_mask(obj.segment)

                if obj.is_one_dau:
                    continue

                # Divided cells are only used for the masks since the mother cells are the inputs to the frame
                if obj.is_two_dau:                 
                    dividing_masks[obj.object_id] = obj.segment.to(torch.bool)
                    dividing_centroids[obj.object_id] = centroid                        
                    continue 

                if (obj.daughter_ids > 0).sum() == 2:
                    step_t_daughter_ids[t].append(obj.daughter_ids)
                elif (obj.daughter_ids > 0).sum() == 1:
                    step_t_daughter_ids[t].append(torch.tensor([obj.daughter_ids[0], 0], dtype=torch.int32))
                else:
                    step_t_daughter_ids[t].append(torch.zeros((2), dtype=torch.int32))

                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int32)
                )

                # Skip the mask of the mother cell dividing since we will use the daugher cells masks instead
                # The mother cell is the input and the daughter cells are the outputs
                if (obj.daughter_ids > 0).sum() == 0:
                    step_t_masks[t].append(obj.segment.to(torch.bool))
                    step_t_centroids[t].append(centroid)
                elif (obj.daughter_ids > 0).sum() == 1:
                    dau_id = obj.daughter_ids[0]
                    dau_obj = next((o for o in objects if o.object_id == dau_id), None)
                    if dau_obj is not None:
                        step_t_masks[t].append(dau_obj.segment.to(torch.bool))
                        step_t_centroids[t].append(get_centroids_from_mask(dau_obj.segment))

                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx], dtype=torch.int32)
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

                step_t_cell_divides[t].append((obj.daughter_ids > 0).sum() == 2)
                # This signifies that a cell is being tracked to the next frame regardless if it exists in the next frame or not
                # This keeps track of cells being tracked after exiting the current frame for VOSSampler.num_frames_track_lost_objects frames
                # The VOS Sampler decides the number of frames we track object after it exits
                step_t_cell_tracks_mask[t].append((obj.is_in_next_object_ids_list))
                step_t_target_obj_mask[t].append(obj.segment.sum() > 0 or (obj.daughter_ids > 0).sum() > 0)

            for daughter_ids in step_t_daughter_ids[t]:
                if (daughter_ids > 0).sum() == 2:
                    for daughter_id in daughter_ids:
                        step_t_masks[t].append(dividing_masks[int(daughter_id)])
                        step_t_centroids[t].append(dividing_centroids[int(daughter_id)])

            if not step_t_obj_to_frame_idx[t]:
                step_t_no_inputs.append(torch.tensor(True))
            else:
                step_t_no_inputs.append(torch.tensor(False))

    # Handle empty lists to prevent stack errors
    for t in range(T):
        if not step_t_obj_to_frame_idx[t]:
            step_t_obj_to_frame_idx[t].append(torch.tensor([t, 0], dtype=torch.int32))
            step_t_masks[t].append(torch.zeros((H, W), dtype=torch.bool))
            step_t_objects_identifier[t].append(torch.tensor([0, 0, 0]))
            step_t_frame_orig_size[t].append(torch.tensor([H, W]))
            step_t_cell_divides[t].append(torch.tensor(False))
            step_t_cell_tracks_mask[t].append(torch.tensor(False))
            step_t_target_obj_mask[t].append(torch.tensor(False))
            step_t_daughter_ids[t].append(torch.zeros((2), dtype=torch.int32))
            step_t_centroids[t].append(torch.zeros((2), dtype=torch.float32))

    # Stack tensors for each time step
    obj_to_frame_idx_per_t = [torch.stack(obj_to_frame_idx, dim=0) for obj_to_frame_idx in step_t_obj_to_frame_idx]
    masks_per_t = [torch.stack(masks, dim=0) for masks in step_t_masks]
    objects_identifier_per_t = [torch.stack(id, dim=0) for id in step_t_objects_identifier]
    frame_orig_size_per_t = [torch.stack(id, dim=0) for id in step_t_frame_orig_size]
    cell_divides_per_t = [torch.stack(id, dim=0) for id in step_t_cell_divides]
    cell_tracks_mask_per_t = [torch.tensor(id, dtype=torch.bool) for id in step_t_cell_tracks_mask]
    target_obj_mask_per_t = [torch.stack(id, dim=0) for id in step_t_target_obj_mask]
    daughter_ids_per_t = [torch.stack(id, dim=0) for id in step_t_daughter_ids]
    centroids_per_t = [torch.stack(id, dim=0) for id in step_t_centroids]
    
    no_inputs = torch.stack(step_t_no_inputs, dim=0) # whether a frame any inputs, foreground or background
    
    # Create heatmaps - these are per time step (not per object), so use original non-padded data
    heatmaps = []
    for t in range(T):
        # Use the actual (non-padded) masks and centroids for this time step
        heatmap = make_gaussian_heatmap(H, W, centroids_per_t[t], masks_per_t[t])
        heatmaps.append(heatmap)
    heatmaps = torch.stack(heatmaps, dim=0)  # Shape: [T, H, W]
    
    # Find maximum number of objects across all time steps
    max_objects = max([t.shape[0] for t in obj_to_frame_idx_per_t]) if obj_to_frame_idx_per_t else 1
    max_objects_masks = max([t.shape[0] for t in masks_per_t]) if masks_per_t else 1
    
    # Create padding mask: True for real objects, False for padded objects
    is_real_per_t = []
    is_real_per_t_masks = []
    for t in range(T):

        is_real_per_t.append(torch.cat([
            torch.ones(obj_to_frame_idx_per_t[t].shape[0], dtype=torch.bool, device=obj_to_frame_idx_per_t[t].device),
            torch.zeros(max_objects - obj_to_frame_idx_per_t[t].shape[0], dtype=torch.bool, device=obj_to_frame_idx_per_t[t].device)
        ]))
        is_real_per_t_masks.append(torch.cat([
            torch.ones(masks_per_t[t].shape[0], dtype=torch.bool, device=masks_per_t[t].device),
            torch.zeros(max_objects_masks - masks_per_t[t].shape[0], dtype=torch.bool, device=masks_per_t[t].device)
        ]))

    is_real = torch.stack(is_real_per_t, dim=0)  # Shape: [T, max_objects]
    is_real_masks = torch.stack(is_real_per_t_masks, dim=0)  # Shape: [T, max_objects_masks]

    # Pad and stack all object-level tensors
    obj_to_frame_idx = pad_and_stack(obj_to_frame_idx_per_t, max_objects, pad_value=0)
    masks = pad_and_stack(masks_per_t, max_objects_masks, pad_value=False)
    objects_identifier = pad_and_stack(objects_identifier_per_t, max_objects, pad_value=0)
    frame_orig_size = pad_and_stack(frame_orig_size_per_t, max_objects, pad_value=0)
    cell_divides = pad_and_stack(cell_divides_per_t, max_objects, pad_value=False)
    cell_tracks_mask = pad_and_stack(cell_tracks_mask_per_t, max_objects, pad_value=False)
    target_obj_mask = pad_and_stack(target_obj_mask_per_t, max_objects, pad_value=False)
    daughter_ids = pad_and_stack(daughter_ids_per_t, max_objects, pad_value=0)
    centroids = pad_and_stack(centroids_per_t, max_objects_masks, pad_value=0.0)

    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        heatmaps=heatmaps,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        bkgd_masks=bkgd_masks,
        cell_divides=cell_divides,
        cell_tracks_mask=cell_tracks_mask,
        target_obj_mask=target_obj_mask,
        daughter_ids=daughter_ids,
        no_inputs=no_inputs,
        dict_key=dict_key,
        is_real=is_real,
        is_real_masks=is_real_masks,
        centroids=centroids,
        batch_size=[T],
    )

def make_gaussian_heatmap(h, w, centers, masks, sigma=3):
    """Returns (H, W) heatmap with Gaussians at each (x, y) center."""
    y = torch.arange(h).view(h, 1).expand(h, w)
    x = torch.arange(w).view(1, w).expand(h, w)
    heatmap = torch.zeros((h, w))
    masks_resized = F.interpolate(masks.unsqueeze(0)*1.0, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
    for (cx, cy), mask_resized in zip(centers, masks_resized):
        if cx == 0 and cy == 0:
            continue
        g = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        g = g * mask_resized
        heatmap = torch.maximum(heatmap, g)  # in case of overlapping cells
    return heatmap

def get_centroids_from_mask(mask):
    """
    Args:
        mask: binary (H, W) tensor

    Returns:
        [cx, cy] tensor (x, y) in pixel coords. If mean is not on the mask, uses middle value xs[len//2], ys[len//2].
    """
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return torch.zeros((2), dtype=torch.float32)
    cx = xs.float().mean()
    cy = ys.float().mean()
    # If mean falls outside the mask (e.g. filament), use middle value: xs[len//2], ys[len//2] so cx, cy are from the mask.
    H, W = mask.shape
    cy_int = int(round(cy.item()))
    cx_int = int(round(cx.item()))
    if not mask[cy_int, cx_int]:
        cx = xs[len(xs) // 2].float()
        cy = ys[len(ys) // 2].float()
    return torch.tensor([cx.item(), cy.item()], dtype=torch.float32)