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

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

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
        """

        flat_idx = []

        for i in range(len(self.obj_to_frame_idx)):
            batch = self.obj_to_frame_idx[i]
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

                # Divided cells are only used for the masks since the mother cells are the inputs to the frame
                if t > 0 and obj.entering and obj.parent_id > 0:                      
                    dividing_masks[obj.object_id] = obj.segment.to(torch.bool)
                    dividing_centroids[obj.object_id] = centroid
                    continue 

                if obj.daughter_ids.sum() > 0:
                    step_t_daughter_ids[t].append(obj.daughter_ids)
                else:
                    step_t_daughter_ids[t].append(torch.zeros((2), dtype=torch.int32))

                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )

                # Skip the mask of the mother cell dividing since we will use the daugher cells masks instead
                # The mother cell is the input and the daughter cells are the outputs
                if obj.daughter_ids.sum() == 0:
                    step_t_masks[t].append(obj.segment.to(torch.bool))
                    step_t_centroids[t].append(centroid)

                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

                step_t_cell_divides[t].append(obj.daughter_ids.sum() > 0)
                # This signifies that a cell is being tracked to the next frame regardless if it exists in the next frame or not
                # This keeps track of cells being tracked after exiting the current frame for VOSSampler.num_frames_track_lost_objects frames
                # The VOS Sampler decides the number of frames we track object after it exits
                step_t_cell_tracks_mask[t].append((obj.is_in_next_object_ids_list))
                step_t_target_obj_mask[t].append(obj.segment.sum() > 0 or obj.daughter_ids.sum() > 0)

            for daughter_ids in step_t_daughter_ids[t]:
                if daughter_ids.sum() > 0:
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
            step_t_obj_to_frame_idx[t].append(torch.tensor([t, 0], dtype=torch.int))
            step_t_masks[t].append(torch.zeros((H, W), dtype=torch.bool))
            step_t_objects_identifier[t].append(torch.tensor([0, 0, 0]))
            step_t_frame_orig_size[t].append(torch.tensor([H, W]))
            step_t_cell_divides[t].append(torch.zeros(1, dtype=torch.bool))
            step_t_cell_tracks_mask[t].append(torch.zeros(1, dtype=torch.bool))
            step_t_target_obj_mask[t].append(torch.zeros(1, dtype=torch.bool))
            step_t_daughter_ids[t].append(torch.zeros((2), dtype=torch.int32))
            step_t_centroids[t].append(torch.zeros((2), dtype=torch.float32))

    obj_to_frame_idx = [torch.stack(obj_to_frame_idx, dim=0) for obj_to_frame_idx in step_t_obj_to_frame_idx]
    masks = [torch.stack(masks, dim=0) for masks in step_t_masks]
    objects_identifier = [torch.stack(id, dim=0) for id in step_t_objects_identifier]
    frame_orig_size = [torch.stack(id, dim=0) for id in step_t_frame_orig_size]
    cell_divides = [torch.stack(id, dim=0) for id in step_t_cell_divides]
    cell_tracks_mask = [torch.tensor(id, dtype=torch.bool) for id in step_t_cell_tracks_mask] # whether the object is being tracked to the next frame regardless if it exists in the current frame
    target_obj_mask = [torch.stack(id, dim=0) for id in step_t_target_obj_mask] # whether the cell exists in the frame
    daughter_ids = [torch.stack(id, dim=0) for id in step_t_daughter_ids]
    no_inputs = torch.stack(step_t_no_inputs, dim=0) # whether a frame any inputs, foreground or background
    
    centroids = [torch.stack(id, dim=0) for id in step_t_centroids]
    heatmaps = [make_gaussian_heatmap(H//4, W//4, centroid/4, mask) for centroid,mask in zip(centroids,masks)]

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
        batch_size=[T],
    )

def make_gaussian_heatmap(h, w, centers, masks, sigma=1):
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
        (x, y) float tuple
    """
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return torch.zeros((2), dtype=torch.float32)
    cx = xs.float().mean()
    cy = ys.float().mean()
    return torch.tensor([cx.item(), cy.item()], dtype=torch.float32)