# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image as PILImage


class CTCSegmentLoader:
    def __init__(self, video_mask_path, first_mask_path, target_size, 
                 resize_threshold, training):
        
        self.mask_paths = sorted(list((video_mask_path).glob("*.tif")))
        self.training = training
        self.target_size = target_size
        self.resize_threshold = resize_threshold
        self.crop_region = self._determine_crop_region(first_mask_path)
        self._frame_crop_cache = {}

    def _determine_crop_region(self, first_mask_path: str) -> Optional[Tuple[int, int, int, int]]:
        """Determine crop region: 10% random, 90% center on random cell for training; always center crop for validation."""
        # Load first frame image to determine size
        first_mask = tifffile.imread(first_mask_path)
        h, w = first_mask.shape
        self.full_h = h
        self.full_w = w
        max_dim = max(h, w)
        
        # Only crop if image is much larger than target
        if max_dim <= self.resize_threshold:
            return None
        
        scale = 1.0 + random.uniform(-0.05, 0.05) if self.training else 1.0
        crop_h = max(1, int(round(self.target_size * scale)))
        crop_w = max(1, int(round(self.target_size * scale)))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        
        # Determine crop position
        # Training: 10% random crop, 90% center on random cell
        if self.training and random.random() < 0.1:
            # Random crop
            top = random.randint(0, max(0, h - crop_h))
            left = random.randint(0, max(0, w - crop_w))
        else:
            # Load first frame mask to find cells
            instance_ids = np.unique(first_mask)
            valid_cells = instance_ids[instance_ids != 0]
            if valid_cells.size > 0:

                # Center crop on random cell
                if self.training:
                    valid_cell_id= random.choice(valid_cells)
                else:
                    valid_cell_id = valid_cells[0]
                where_cell = np.where(first_mask == valid_cell_id)
                h_cell, w_cell = int(np.median(where_cell[0])), int(np.median(where_cell[1]))
                top = h_cell - crop_h // 2
                left = w_cell - crop_w // 2
            
        # Ensure we don't go out of bounds
        top = min(max(0,top), max(0, h - crop_h))
        left = min(max(0,left), max(0, w - crop_w))
        bottom = min(top + crop_h, h)
        right = min(left + crop_w, w)
        
        return (top, left, bottom, right)

    def _get_frame_crop_region(self, frame_id):
        if self.crop_region is None:
            return None
        if frame_id in self._frame_crop_cache:
            return self._frame_crop_cache[frame_id]
        if not self.training or random.random() >= 0.5:
            self._frame_crop_cache[frame_id] = self.crop_region
            return self.crop_region
        top, left, bottom, right = self.crop_region
        crop_h = bottom - top
        crop_w = right - left
        max_shift_y = min(50, int(round(0.05 * crop_h)))
        max_shift_x = min(50, int(round(0.05 * crop_w)))
        shift_y = random.randint(-max_shift_y, max_shift_y)
        shift_x = random.randint(-max_shift_x, max_shift_x)
        new_top = min(max(0, top + shift_y), max(0, self.full_h - crop_h))
        new_left = min(max(0, left + shift_x), max(0, self.full_w - crop_w))
        crop_region = (new_top, new_left, new_top + crop_h, new_left + crop_w)
        self._frame_crop_cache[frame_id] = crop_region
        return crop_region

    def load(self, frame_id):
        """
        Mimics SAM2 segment loaders by returning a dictionary of binary masks.
        """
        mask_path = self.mask_paths[frame_id]
        mask = tifffile.imread(mask_path)

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]

        segments = {}
        crop_region = self._get_frame_crop_region(frame_id)
        for inst_id in instance_ids:
            segment = torch.from_numpy(mask == inst_id)
            
            # Apply crop if needed
            if crop_region is not None:
                top, left, bottom, right = crop_region
                segment = segment[top:bottom, left:right]

            # Assume 
            if segment.sum() > 10:
                segments[int(inst_id)] = segment

        # Dilate background mask to avoid points touching objects to ensure there is no confusion between FPs being interpreted as objects
        kernel = np.ones((3,3), np.uint8)
        bkgd_mask_dilated = cv2.erode((mask == 0).astype(np.uint8), kernel, iterations=2) # Erode background = dilate objects
        bkgd_mask = torch.from_numpy(bkgd_mask_dilated.astype(bool))
        
        # Apply crop to background mask if needed
        if crop_region is not None:
            top, left, bottom, right = crop_region
            bkgd_mask = bkgd_mask[top:bottom, left:right]
        
        segments['bkgd_mask'] = bkgd_mask

        return segments

class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids
        
        try:
            from pycocotools import mask as mask_utils
            self.mask_utils = mask_utils
        except ImportError:
            raise ImportError("pycocotools is required for JSONSegmentLoader")
            
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])
                    else:
                        annotations_fps = int(data["fps"])
                    assert frames_fps % annotations_fps == 0
                    self.ann_every = frames_fps // annotations_fps
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        assert frame_id % self.ann_every == 0
        rle_mask = self.frame_annots[frame_id // self.ann_every]

        valid_objs_ids = set(range(len(rle_mask)))
        if self.valid_obj_ids is not None:
            # Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # Decode the masks
        raw_segments = torch.from_numpy(self.mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]
        return segments

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        res = {obj_id: [] for obj_id in range(num_objects)}

        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        video_png_root: the folder contains all the masks stored in png
        """
        self.video_png_root = video_png_root
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        png_filenames = os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            frame_id, _ = os.path.splitext(filename)
            self.frame_id_to_png_filename[int(frame_id)] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # load the mask
        masks = PILImage.open(mask_path).convert("P")
        masks = np.array(masks)

        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # remove background (0)

        # convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]
        self.W = tmp_mask.shape[1]
        if self.single_object_mode:
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError("pycocotools is required for LazySegments")
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        return self.segments
