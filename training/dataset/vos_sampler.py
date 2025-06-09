# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids_list: List[List[int]]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class FrameIndexSampler(VOSSampler):
    """
    Sampler that handles object selection for frames.
    Frame selection is handled by the DataLoader's sampler using the dataset's frame_index.
    For training: randomly samples up to max_num_objects
    For validation: takes first max_num_objects objects
    """
    def __init__(
        self,
        max_num_objects,
        max_num_bkgd_objects,
        is_training,
        num_frames_track_lost_objects=1,
    ):
        super().__init__(sort_frames=not is_training)  # For val, we want sorted frames
        self.max_num_objects = max_num_objects
        self.max_num_bkgd_objects = max_num_bkgd_objects
        self.is_training = is_training
        self.num_frames_track_lost_objects = num_frames_track_lost_objects

    def sample(self, video, segment_loader, epoch=None):
        """
        Handle object selection for the provided frames.
        Frames are already selected by the DataLoader's sampler using dataset's frame_index.
        For training: randomly samples up to max_num_objects
        For validation: takes first max_num_objects objects
        """
        frames = video.frames

        # Get first frame object ids
        visible_object_ids = []
        loaded_segms = segment_loader.load(frames[0].frame_idx)
        if isinstance(loaded_segms, LazySegments):
            visible_object_ids = list(loaded_segms.keys())
        else:
            for object_id, segment in segment_loader.load(
                frames[0].frame_idx
            ).items():
                if segment.sum() and object_id != 'bkgd_mask':
                    visible_object_ids.append(object_id)

        # Sample objects based on mode
        if self.is_training:
            # Random sample for training
            object_ids = sorted(random.sample(
                visible_object_ids,
                min(len(visible_object_ids), self.max_num_objects),
            ))
        else:
            # Take first N objects for validation
            object_ids = sorted(visible_object_ids)[:self.max_num_objects]

        object_ids_list = [object_ids]
        object_ids_dict = {0: object_ids}

        # Handle object tracking if needed
        if video.man_track is not None and len(frames) > 1:
            for i, frame in enumerate(frames):
                if i == 0:
                    continue
                
                # Get all object ids in the current frame
                object_ids_dict[i] = []
                input_object_ids = []
                
                for object_id, segment in segment_loader.load(frame.frame_idx).items():
                    if isinstance(object_id, int):
                        parent_id = video.man_track[video.man_track[:, 0] == object_id, -1]
                        if any([object_id in object_ids_dict[j] for j in range(i)]) or any([parent_id in object_ids_dict[j] for j in range(i)]):
                            input_object_ids.append(object_id)
                            object_ids_dict[i].append(object_id)
                
                # Include objects from previous frames within tracking window
                for j in range(i-self.num_frames_track_lost_objects, i):
                    if j >= 0:  # Ensure we don't access negative indices
                        input_object_ids.extend(object_ids_dict[j])
                
                # Remove duplicates and sort
                input_object_ids = sorted(list(set(input_object_ids)))
                object_ids_list.append(input_object_ids)
        else:
            # If no tracking or single frame, use same objects for all frames
            object_ids_list.extend([object_ids] * (len(frames) - 1))

        # Add background points for training
        if self.is_training:
            max_num_bkgd_points = min(
                max(0, self.max_num_objects - len(object_ids_list[0])), 
                self.max_num_bkgd_objects
            )
            min_num_bkgd_points = int(len(object_ids_list[0]) == 0)
            num_bkgd_points = random.randint(min_num_bkgd_points, max_num_bkgd_points)

            # Generate background object IDs using negative integers
            bkgd_object_ids = list(range(-1, -1 - num_bkgd_points, -1))

            # Add background objects to frames within tracking window
            for j in range(0, min(len(object_ids_list), self.num_frames_track_lost_objects + 1)):
                object_ids_list[j] = object_ids_list[j] + bkgd_object_ids

        return SampledFramesAndObjects(frames=frames, object_ids_list=object_ids_list)
