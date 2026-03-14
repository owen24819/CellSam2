# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from sam2.utils.misc import read_image
from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.utils.data_utils import Frame, Object, VideoDatapoint


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_size=512,
        ):

        self._transforms = transforms
        self.video_dataset = video_dataset
        self.sampler = sampler
        self.target_size = target_size

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to video_dataset (e.g. for CTCRawDataset per-epoch caps)."""
        self.curr_epoch = epoch
        if hasattr(self.video_dataset, "set_epoch"):
            self.video_dataset.set_epoch(epoch)

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
        Constructs a VideoDatapoint sample to pass to transforms.
        Only tracks cells that appear in the first frame. If a cell leaves and comes back,
        it is not tracked.
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids_list = sampled_frms_and_objs.object_ids_list
        man_track = video.man_track

        images = []
        crop_regions = [
            segment_loader._get_frame_crop_region(frame.frame_idx)
            for frame in sampled_frames
        ]
        rgb_images = load_images(sampled_frames, crop_regions)
        # After crop, images are crop_size (may be non-square), otherwise original size
        if crop_regions and crop_regions[0] is not None:
            top, left, bottom, right = crop_regions[0]
            final_size = (bottom - top, right - left)
        else:
            final_size = rgb_images[0].size[::-1]
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
                is_two_dau = False
                is_one_dau = False
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

                        if entering and parent_id > 0:
                            # Check if parent has exactly 2 daughters (count occurrences of parent_id in last column)
                            if (man_track[:,-1] == parent_id).sum() == 2:
                                is_two_dau = True
                            elif (man_track[:,-1] == parent_id).sum() == 1:
                                is_one_dau = True
                
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
                        is_two_dau=is_two_dau,
                        is_one_dau=is_one_dau,
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
            size=final_size,
            man_track=man_track,
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames, crop_regions=None):
    all_images = []
    if crop_regions is None:
        crop_regions = [None] * len(frames)
    for frame, crop_region in zip(frames, crop_regions, strict=False):
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            image = read_image(path)
            if crop_region is not None:
                top, left, bottom, right = crop_region
                image = image.crop((left, top, right, bottom))
            all_images.append(image)
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
