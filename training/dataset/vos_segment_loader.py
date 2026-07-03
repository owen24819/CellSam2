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
    """Load CTC masks with optional spatial crop and clip-consistent geometry.

    ``zoom_range`` is sampled once per clip when ``zoom_p`` triggers. When the source
    image is larger than ``resize_threshold``, zoom-in uses a smaller crop window;
    zoom-out uses a larger crop window when the source is big enough. When the crop
    cannot reach the full window, the remaining factor is applied via warp zoom
    (multiplicative: ``crop_zoom * warp_zoom == zoom_scale``).
    """

    def __init__(
        self,
        video_mask_path,
        first_mask_path,
        target_size,
        resize_threshold,
        training,
        *,
        crop_shift_frac: float = 0.08,
        crop_shift_p: float = 0.5,
        random_crop_p: float = 0.1,
        zoom_range=(0.6, 1.2),
        zoom_p: float = 0.5,
        max_shear_deg: float = 45.0,
        shear_p: float = 0.3,
        wide_aspect_ratio_min: float = 3.0,
        wide_aspect_shear_cap: float = 10.0,
        aniso_scale_range=(1.0, 4.0),
        aniso_p: float = 0.3,
    ):
        self.mask_paths = sorted(list((video_mask_path).glob("*.tif")))
        self.training = training
        self.target_size = target_size
        self.resize_threshold = resize_threshold
        self.crop_shift_frac = crop_shift_frac
        self.crop_shift_p = crop_shift_p
        self.random_crop_p = random_crop_p
        self.zoom_range = tuple(zoom_range)
        self.zoom_p = zoom_p
        self.max_shear_deg = max_shear_deg
        self.shear_p = shear_p
        self.wide_aspect_ratio_min = wide_aspect_ratio_min
        self.wide_aspect_shear_cap = wide_aspect_shear_cap
        self.aniso_scale_range = tuple(aniso_scale_range)
        self.aniso_p = aniso_p
        self._frame_crop_cache = {}

        first_mask = tifffile.imread(first_mask_path)
        self.full_h, self.full_w = first_mask.shape
        self.effective_max_shear_deg = self._resolve_effective_max_shear(
            self.full_h, self.full_w
        )

        self._sample_geometry()
        self.crop_region = self._determine_crop_region(first_mask)
        self.warp_zoom_scale = self._resolve_warp_zoom_scale()

    def _resolve_warp_zoom_scale(self) -> float:
        """Warp zoom after any crop contribution (``crop_zoom * warp_zoom == zoom_scale``)."""
        if self.zoom_scale == 1.0:
            return 1.0
        if self.crop_region is None:
            return self.zoom_scale

        top, left, bottom, right = self.crop_region
        crop_h = bottom - top
        crop_w = right - left
        actual_crop = max(1, min(crop_h, crop_w))
        crop_zoom = self.target_size / actual_crop
        return self.zoom_scale / crop_zoom

    @staticmethod
    def _aspect_ratio(h: int, w: int) -> float:
        return max(h, w) / max(1, min(h, w))

    def _resolve_effective_max_shear(self, h: int, w: int) -> float:
        """Cap shear on wide movies where large angles look unrealistic."""
        if self._aspect_ratio(h, w) >= self.wide_aspect_ratio_min:
            return min(self.max_shear_deg, self.wide_aspect_shear_cap)
        return self.max_shear_deg

    def _determine_crop_region(self, first_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Determine crop region: 10% random, 90% center on random cell for training; always center crop for validation."""
        h, w = first_mask.shape
        max_dim = max(h, w)
        
        # Only crop if image is much larger than target
        if max_dim <= self.resize_threshold:
            return None

        crop_h = max(1, int(round(self.target_size / self.zoom_scale)))
        crop_w = max(1, int(round(self.target_size / self.zoom_scale)))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        
        # Determine crop position
        # Training: 10% random crop, 90% center on random cell
        if self.training and random.random() < self.random_crop_p:
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
        if not self.training or random.random() >= self.crop_shift_p:
            self._frame_crop_cache[frame_id] = self.crop_region
            return self.crop_region
        top, left, bottom, right = self.crop_region
        crop_h = bottom - top
        crop_w = right - left
        max_shift_y = int(round(self.crop_shift_frac * crop_h))
        max_shift_x = int(round(self.crop_shift_frac * crop_w))
        shift_y = random.randint(-max_shift_y, max_shift_y)
        shift_x = random.randint(-max_shift_x, max_shift_x)
        new_top = min(max(0, top + shift_y), max(0, self.full_h - crop_h))
        new_left = min(max(0, left + shift_x), max(0, self.full_w - crop_w))
        crop_region = (new_top, new_left, new_top + crop_h, new_left + crop_w)
        self._frame_crop_cache[frame_id] = crop_region
        return crop_region

    def _sample_geometry(self):
        """Sample zoom / shear / anisotropic stretch once per clip (training only)."""
        if not self.training:
            self.zoom_scale = 1.0
            self.shear_deg = 0.0
            self.aniso_axis = None
            self.aniso_scale = 1.0
            return

        if self.zoom_p > 0 and random.random() < self.zoom_p:
            lo, hi = self.zoom_range
            self.zoom_scale = random.uniform(lo, hi)
        else:
            self.zoom_scale = 1.0

        if self.effective_max_shear_deg > 0 and self.shear_p > 0 and random.random() < self.shear_p:
            self.shear_deg = random.uniform(
                -self.effective_max_shear_deg, self.effective_max_shear_deg
            )
        else:
            self.shear_deg = 0.0

        if self.aniso_p > 0 and random.random() < self.aniso_p:
            self.aniso_axis = random.choice([0, 1])  # 0=y (height), 1=x (width)
            lo, hi = self.aniso_scale_range
            self.aniso_scale = random.uniform(lo, hi)
        else:
            self.aniso_axis = None
            self.aniso_scale = 1.0

    def _geometry_is_identity(self) -> bool:
        return (
            self.warp_zoom_scale == 1.0
            and self.shear_deg == 0.0
            and (self.aniso_axis is None or self.aniso_scale == 1.0)
        )

    @staticmethod
    def _resize_center_crop(
        arr: np.ndarray, new_h: int, new_w: int, out_h: int, out_w: int, interp, fill
    ) -> np.ndarray:
        """Resize to (new_h, new_w), then center-crop/pad back to (out_h, out_w)."""
        arr = cv2.resize(arr, (new_w, new_h), interpolation=interp)
        cur_h, cur_w = arr.shape[:2]
        top = max(0, (cur_h - out_h) // 2)
        left = max(0, (cur_w - out_w) // 2)
        arr = arr[top : top + min(out_h, cur_h), left : left + min(out_w, cur_w)]

        cur_h, cur_w = arr.shape[:2]
        pad_h = out_h - cur_h
        pad_w = out_w - cur_w
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            arr = cv2.copyMakeBorder(
                arr,
                pad_top,
                pad_h - pad_top,
                pad_left,
                pad_w - pad_left,
                cv2.BORDER_CONSTANT,
                value=fill,
            )
        return arr

    @staticmethod
    def apply_isotropic_zoom(arr: np.ndarray, zoom_scale: float, is_mask: bool) -> np.ndarray:
        """Resize by ``zoom_scale``, then center-crop/pad back to the original size."""
        if zoom_scale == 1.0:
            return arr
        h, w = arr.shape[:2]
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        new_h = max(1, int(round(h * zoom_scale)))
        new_w = max(1, int(round(w * zoom_scale)))
        return CTCSegmentLoader._resize_center_crop(
            arr, new_h, new_w, h, w, interp, fill=0
        )

    def _warp_array(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        """Apply clip-consistent zoom, shear, then anisotropic stretch. Keeps HxW size."""
        if self._geometry_is_identity():
            return arr

        h, w = arr.shape[:2]
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        fill = 0

        # Remaining zoom after crop window (multiplicative split).
        if self.warp_zoom_scale != 1.0:
            arr = self.apply_isotropic_zoom(arr, self.warp_zoom_scale, is_mask)

        if self.shear_deg != 0.0:
            shear = np.tan(np.radians(self.shear_deg))
            # Horizontal shear about image center (matches torchvision shear-x).
            cy = h / 2.0
            M = np.array([[1.0, shear, -shear * cy], [0.0, 1.0, 0.0]], dtype=np.float32)
            arr = cv2.warpAffine(
                arr,
                M,
                (w, h),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill,
            )

        if self.aniso_axis is not None and self.aniso_scale != 1.0:
            scale = self.aniso_scale
            if self.aniso_axis == 1:  # stretch width (x)
                new_w = max(1, int(round(w * scale)))
                new_h = h
            else:  # stretch height (y)
                new_w = w
                new_h = max(1, int(round(h * scale)))
            arr = self._resize_center_crop(arr, new_h, new_w, h, w, interp, fill)

        return arr

    def apply_geometry_mask(self, segment: torch.Tensor) -> torch.Tensor:
        """Warp a binary mask with the clip-consistent geometry."""
        if self._geometry_is_identity():
            return segment
        arr = segment.detach().cpu().numpy().astype(np.uint8)
        arr = self._warp_array(arr, is_mask=True)
        return torch.from_numpy(arr.astype(bool))

    def apply_geometry_image(self, image: PILImage.Image) -> PILImage.Image:
        """Warp a PIL image with the clip-consistent geometry."""
        if self._geometry_is_identity():
            return image
        arr = np.array(image)
        arr = self._warp_array(arr, is_mask=False)
        return PILImage.fromarray(arr)

    def prepare_image(self, image: PILImage.Image, frame_id) -> PILImage.Image:
        """Crop (if any) then apply clip-consistent geometry."""
        crop_region = self._get_frame_crop_region(frame_id)
        if crop_region is not None:
            top, left, bottom, right = crop_region
            image = image.crop((left, top, right, bottom))
        return self.apply_geometry_image(image)

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

            segment = self.apply_geometry_mask(segment)

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

        bkgd_mask = self.apply_geometry_mask(bkgd_mask)

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
