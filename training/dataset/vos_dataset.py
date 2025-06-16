# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint
class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target

    def _get_datapoint(self, idx):

        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        # sample a video
        video, segment_loader = self.video_dataset.get_video(idx)
        # sample frames and object indices to be used in a datapoint
        sampled_frms_and_objs = self.sampler.sample(
            video, segment_loader, epoch=self.curr_epoch
        )

        datapoint = self.construct(video, sampled_frms_and_objs, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids_list = sampled_frms_and_objs.object_ids_list
        man_track = video.man_track

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, (frame, sampled_object_ids) in enumerate(zip(sampled_frames, sampled_object_ids_list)):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                    object_ids=sampled_object_ids,
                )
            )
            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = segment_loader.load(frame.frame_idx)

            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                # Initialize default values
                parent_id = 0
                entering = True
                daughter_ids = torch.zeros((2), dtype=torch.int32)
                is_in_next_object_ids_list = True

                if man_track is not None and obj_id > 0:
                    # Get cell lineage information from man_track
                    cell_info = man_track[man_track[:,0] == obj_id]
                    if len(cell_info) > 0:  # Check if cell_info is not empty
                        cell_info = cell_info[0]
                        _, start_frame, end_frame, parent_id = cell_info
                        parent_id = int(parent_id)

                        # Cell is entering if current frame is its start frame
                        entering = bool(start_frame == frame.frame_idx)

                        # Check if this cell has daughter cells and is currently dividing
                        if obj_id in man_track[:,-1] and end_frame + 1 == frame.frame_idx:
                            # Get IDs of daughter cells when division occurs
                            daughter_ids = torch.tensor(
                                man_track[man_track[:,-1] == obj_id, 0], 
                                dtype=torch.int32
                            )
                
                # Determine if this cell should be tracked in the next frame
                if frame_idx < len(sampled_object_ids_list) - 1:
                    # Cell is in next frame's object list or has daughter cells
                    is_in_next_object_ids_list = (
                        obj_id in sampled_object_ids_list[frame_idx+1] or 
                        daughter_ids.sum() > 0
                    )

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                        entering=entering,
                        parent_id=parent_id,
                        daughter_ids=daughter_ids,
                        is_in_next_object_ids_list=is_in_next_object_ids_list
                    )
                )

            # Add background mask if available
            if 'bkgd_mask' in segments:
                images[frame_idx].objects.append(
                    Object(
                        object_id=-1000,
                        frame_index=frame.frame_idx,
                        segment=segments['bkgd_mask'],
                    )
                )
            
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
            man_track=man_track,
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                image = PILImage.open(fopen)

                if image.mode == "RGB":
                    pass
                elif image.mode == "I;16":
                    # Convert to NumPy array
                    arr = np.array(image)

                    # Get 1st and 99th percentiles for robust scaling
                    p1, p99 = np.percentile(arr, [1, 99])
                    
                    # Handle case where all values are identical (including all zeros)
                    if p1 == p99:
                        arr_8bit = np.zeros_like(arr, dtype=np.uint8)
                    else:
                        # Clip values to percentiles and scale to 0-255
                        arr_clipped = np.clip(arr, p1, p99)
                        arr_8bit = ((arr_clipped - p1) * (255.0 / (p99 - p1))).astype(np.uint8)

                    # Convert to RGB by stacking
                    image = PILImage.fromarray(arr_8bit).convert("RGB")
                else:
                    raise ValueError(f"Unexpected image mode: {image.mode}. Please inspect and handle this mode manually.")

                all_images.append(image)
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
