# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

import numpy as np

MAX_RETRIES = 1000
@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids_list: List[List[int]]
    man_track: np.ndarray


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
        for object_id, segment in segment_loader.load(frames[0].frame_idx).items():
            if segment is not None and segment.sum() and object_id != 'bkgd_mask':
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

        new_man_track = np.zeros((0,4), dtype=np.int16)
        first_frame_index = frames[0].frame_idx
        for object_id in object_ids:
            new_cell_row = np.array([[object_id, first_frame_index, first_frame_index, 0]], dtype=np.int16)
            new_man_track = np.vstack((new_man_track, new_cell_row))

        # Handle object tracking if needed
        if video.man_track is not None and len(frames) > 1:

            lost_object_ids_dict = {i: [] for i in range(len(frames))}

            for i, frame in enumerate(frames):
                if i == 0:
                    continue
                
                # Get all object ids in the current frame
                object_ids_dict[i] = []
                input_object_ids = []
                daus = {}
                parents = []
                
                for object_id, segment in segment_loader.load(frame.frame_idx).items():
                    if isinstance(object_id, int):
                        parent_id = int(video.man_track[video.man_track[:, 0] == object_id, -1][0])

                        if parent_id > 0:
                            parents.append(parent_id)
                        
                        # Filter to only include: (1) cells that were present in the previous frame (tracked),
                        # or (2) daughter cells whose parent was in the previous frame (from division or budding)
                        if any([object_id in object_ids_dict[i-1]]) or any([parent_id in object_ids_dict[i-1]]):
                            object_ids_dict[i].append(object_id)

                            # Update man_track if cell tracks to next frame
                            if any([object_id in object_ids_dict[i-1]]):
                                input_object_ids.append(object_id)
                                new_man_track[new_man_track[:, 0] == object_id, 2] = frame.frame_idx
                            
                            # Update man_track if cell in previous frame divides with at least one daughter cell in current frame
                            if any([parent_id in object_ids_dict[i-1]]):
                                new_cell_row = np.array([object_id, frame.frame_idx, frame.frame_idx, parent_id], dtype=np.int16)
                                new_man_track = np.vstack((new_man_track, new_cell_row))

                                daus.setdefault(parent_id, []).append(object_id)

                lost_object_ids_dict[i-1] = [obj_id for obj_id in object_ids_dict[i-1] if obj_id not in object_ids_dict[i] and obj_id not in parents]
                                
                # Include objects from previous frames within tracking window
                for j in range(max(0, i-self.num_frames_track_lost_objects), i):
                    input_object_ids.extend(lost_object_ids_dict[j])

                # Post-process daus: find all parent_ids with a single daughter (len==1), collect these, and remove them from daus
                assert all(len(obj_ids) == 1 or len(obj_ids) == 2 for obj_ids in daus.values()), "Each parent_id must have either 1 or 2 daughters"
                single_dau_items = {parent_id: obj_ids[0] for parent_id, obj_ids in daus.items() if len(obj_ids) == 1}
                two_dau_items = {parent_id: obj_ids for parent_id, obj_ids in daus.items() if len(obj_ids) == 2}

                # Optionally: now daus contains only those with 2 daughters
                daus = two_dau_items
                
                # Insert single daughters next to their parent_ids in the ordered list
                previous_ids_ordered = []
                for obj_id in object_ids_list[i-1]:
                    previous_ids_ordered.append(obj_id)
                    if obj_id in single_dau_items:
                        previous_ids_ordered.append(single_dau_items[obj_id])

                # data_utils.py needs both the mother and daughter cells when processing inputs
                # If a cell exits the FOV, then we add it ot hte lost_object_ids_dict where it can be added back for self.num_frames_track_lost_objects frames
                # This use case is for when an cell may disappear then reappear. Also we can further train the model extra frames that the cell is not in the FOV
                ordered_ids = []
                for obj_id in previous_ids_ordered:
                    if obj_id in input_object_ids or obj_id in two_dau_items.keys() or obj_id in single_dau_items.values() or obj_id in single_dau_items.keys():
                        ordered_ids.append(obj_id)

                # Create div_ids from daus dict, ordered by parent_id (sorted parents, sorted daughters within each group)
                div_ids = [dau_id for parent_id in sorted(daus.keys()) for dau_id in sorted(daus[parent_id])]
                
                ordered_input_object_ids = ordered_ids + div_ids
                
                object_ids_list.append(ordered_input_object_ids)

            video.man_track = new_man_track
        else:
            # If no tracking or single frame, use same objects for all frames
            object_ids_list.extend([object_ids] * (len(frames) - 1))

        # Add background points for training
        min_num_bkgd_points = int(len(object_ids_list[0]) == 0)
        num_bkgd_points = random.randint(min_num_bkgd_points, self.max_num_bkgd_objects)

        # Generate background object IDs using negative integers
        bkgd_object_ids = list(range(-1, -1 - num_bkgd_points, -1))

        # Add background objects to frames within tracking window
        for j in range(0, min(len(object_ids_list), self.num_frames_track_lost_objects + 1)):
            object_ids_list[j] = object_ids_list[j] + bkgd_object_ids

        # Only keep frames where objects are being tracked
        # Discard frames where nothing is being tracked (even lost / bkgd objects)
        keep = [idx for idx, object_ids in enumerate(object_ids_list) if len(object_ids) > 0]        

        frames = [frames[i] for i in keep]
        object_ids_list = [object_ids_list[i] for i in keep]

        return SampledFramesAndObjects(frames=frames, object_ids_list=object_ids_list, man_track=new_man_track)
