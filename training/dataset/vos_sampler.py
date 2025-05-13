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


class RandomUniformSampler(VOSSampler):
    """
    VOS Sampler for training: sampling random frames and objects
    """
    def __init__(
        self,
        num_frames,
        max_num_objects,
        max_num_bkgd_objects,
        reverse_time_prob=0.0,
        num_frames_track_lost_objects=1,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.max_num_bkgd_objects = max_num_bkgd_objects
        self.reverse_time_prob = reverse_time_prob

        if num_frames == 1:
            self.num_frames_track_lost_objects = 0
        else:
            self.num_frames_track_lost_objects = num_frames_track_lost_objects

    def sample(self, video, segment_loader, epoch=None):
        """
        Sample random frames and objects from a video

        Args:
            video: The video (VOSVideo) to sample from
            segment_loader: The segment loader (CTCSegmentLoader) to load segments from
            epoch: The epoch number

        Returns:
            SampledFramesAndObjects: The sampled frames and objects
        """

        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum() and object_id != 'bkgd_mask':
                        visible_object_ids.append(object_id)

            object_ids = sorted(random.sample(
                visible_object_ids,
                min(len(visible_object_ids), self.max_num_objects),
            ))

            object_ids_list = [object_ids]
            object_ids_dict = {0: object_ids}

            if video.man_track is not None:
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

            # Calculate how many background points to add
            max_num_bkgd_points = min(
                max(0, self.max_num_objects - len(object_ids_list[0])), 
                self.max_num_bkgd_objects
            )
            num_bkgd_points = random.randint(0, max_num_bkgd_points)

            # Generate background object IDs using negative integers
            bkgd_object_ids = list(range(-1, -1 - num_bkgd_points, -1))  # e.g. [-1, -2, -3, ...]

            # Add background objects to frames within tracking window
            for j in range(0, min(len(object_ids_list), self.num_frames_track_lost_objects + 1)):
                object_ids_list[j] = object_ids_list[j] + bkgd_object_ids

            return SampledFramesAndObjects(frames=frames, object_ids_list=object_ids_list)

        raise Exception("Exceeded MAX_RETRIES in RandomUniformSampler")


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
