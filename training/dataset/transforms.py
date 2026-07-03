# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import random
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import (
    Image as PILImage,
    ImageFilter,
)
from scipy import interpolate
from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint


def hflip(datapoint, index):

    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].data.size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        return datapoint

    def transform_datapoint(self, datapoint: VideoDatapoint):
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


class RandomAnisotropicScale:
    """Stretch the image/masks along one axis (filament-like elongation).

    With probability ``p``, scales one axis by a factor in ``scale_range`` and
    leaves the other at 1.0, then center-crops/pads back to the original size.
    """

    def __init__(
        self,
        scale_range=(1.0, 2.5),
        p=0.3,
        consistent_transform=True,
        image_mean=(123, 116, 103),
        image_interpolation="bilinear",
        axis=None,
        scale=None,
    ):
        """
        Args:
            scale_range: (min, max) stretch factor along the chosen axis.
            p: Probability of applying the transform.
            consistent_transform: Same stretch for all frames in the video.
            image_mean: Fill value for image padding (RGB).
            image_interpolation: ``bilinear`` or ``bicubic`` for images.
            axis: If set (0=y/height, 1=x/width), use this axis (for debugging).
            scale: If set, use this stretch factor (for debugging).
        """
        self.scale_range = scale_range
        self.p = p
        self.consistent_transform = consistent_transform
        self.fill_img = image_mean
        self.axis = axis
        self.scale = scale
        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def _sample_params(self):
        if self.scale is not None:
            scale = float(self.scale)
        else:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
        if self.axis is not None:
            axis = int(self.axis)
        else:
            axis = random.choice([0, 1])
        return axis, scale

    @staticmethod
    def _stretch_and_crop(img, axis, scale, interpolation, fill):
        """Stretch along ``axis`` by ``scale``, then center-crop/pad to original size."""
        if isinstance(img, PILImage.Image):
            w, h = img.size
        else:
            # tensor CHW
            h, w = img.shape[-2:]

        if axis == 1:  # stretch width (x)
            new_w = max(1, int(round(w * scale)))
            new_h = h
        else:  # stretch height (y)
            new_w = w
            new_h = max(1, int(round(h * scale)))

        img = F.resize(img, [new_h, new_w], interpolation=interpolation)

        # Center crop if larger, pad if smaller
        if isinstance(img, PILImage.Image):
            cur_w, cur_h = img.size
        else:
            cur_h, cur_w = img.shape[-2:]

        # Crop
        top = max(0, (cur_h - h) // 2)
        left = max(0, (cur_w - w) // 2)
        img = F.crop(img, top, left, min(h, cur_h), min(w, cur_w))

        if isinstance(img, PILImage.Image):
            cur_w, cur_h = img.size
        else:
            cur_h, cur_w = img.shape[-2:]

        pad_h = h - cur_h
        pad_w = w - cur_w
        if pad_h > 0 or pad_w > 0:
            padding = [
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            ]
            img = F.pad(img, padding, fill=fill)

        return img

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.scale is None and self.axis is None and random.random() >= self.p:
            return datapoint

        if self.consistent_transform:
            axis, scale = self._sample_params()

        for img in datapoint.frames:
            if not self.consistent_transform:
                axis, scale = self._sample_params()

            for obj in img.objects:
                if obj.segment is not None:
                    seg = obj.segment
                    if not isinstance(seg, torch.Tensor):
                        continue
                    # F.resize expects CHW for tensors
                    if seg.dim() == 2:
                        seg = seg.unsqueeze(0)
                    seg = self._stretch_and_crop(
                        seg.float(),
                        axis,
                        scale,
                        InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    obj.segment = seg.squeeze(0) > 0.5

            img.data = self._stretch_and_crop(
                img.data,
                axis,
                scale,
                self.image_interpolation,
                fill=self.fill_img,
            )

        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint

class PadToSquareAPI:
    def __init__(self, size):
        self.size = size

    def __call__(self, datapoint, **kwargs):
        for i in range(len(datapoint.frames)):
            h, w = datapoint.frames[i].size
            
            # Calculate padding
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            
            # Calculate padding on each side to center the image
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            # Pad on all sides to center the image
            padding = [pad_left, pad_top, pad_right, pad_bottom]  # left, top, right, bottom
            datapoint = pad(datapoint, i, padding)
            
        return datapoint
class RandomGaussianBlur:
    def __init__(self, p=0.4, sigma=[0.1,1.5]):
        self.p = p
        self.sigma = sigma

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if random.random() < self.p:
                radius = random.uniform(self.sigma[0],self.sigma[1])
                img.data = self.gaussian_blur(img.data, radius)
        return datapoint

    def gaussian_blur(self, img, radius):
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img



class RandomGaussianNoise:
    def __init__(self, p=0.4, sigma=0.05):
        self.p = p
        self.sigma = sigma

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if random.random() < self.p:
                sigma = random.random() * self.sigma
                img.data = self.gaussian_noise(img.data, sigma)
        return datapoint

    def gaussian_noise(self, img: PILImage.Image, sigma: float) -> PILImage.Image:
        """Apply Gaussian noise to a PIL Image.
        
        Args:
            img: Input PIL Image
            sigma: Standard deviation of the Gaussian noise (0-1 range)
            
        Returns:
            PIL Image with added noise
        """
        # Convert to numpy array preserving all channels
        img_array = np.array(img)
        
        # Generate noise for each channel
        noise = np.random.normal(0, sigma * 255, img_array.shape).astype(np.float32)
        
        # Add noise and clip to valid range
        noisy_img = img_array.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        return PILImage.fromarray(noisy_img)
    


class RandomIlluminationVoodoo:
    def __init__(self, p=0.4, num_control_points=5):
        self.p = p
        self.num_control_points = num_control_points

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.illumination_voodoo(img.data)
        return datapoint
    
    def illumination_voodoo(self, img: PILImage.Image) -> PILImage.Image:
        """Apply random illumination variation to a PIL Image.
        
        Args:
            img: Input PIL Image
            num_control_points: Number of control points for the illumination curve
            
        Returns:
            PIL Image with varied illumination
        """
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Create a random curve along the length of the image
        control_points = np.linspace(0, img_array.shape[0] - 1, num=self.num_control_points)
        random_points = np.random.uniform(low=0.1, high=0.9, size=self.num_control_points)
        mapping = interpolate.PchipInterpolator(control_points, random_points)
        curve = mapping(np.linspace(0, img_array.shape[0] - 1, img_array.shape[0]))
        
        # Reshape curve for multiplication with all channels
        curve_reshaped = np.reshape(curve, (-1, 1, 1) if len(img_array.shape) == 3 else (-1, 1))
        
        # Apply illumination variation to all channels at once
        modified = np.multiply(img_array, curve_reshaped)
        
        # Handle each channel separately for rescaling
        if len(modified.shape) == 3:
            # For RGB images
            result = np.zeros_like(modified)
            for c in range(modified.shape[2]):
                min_val = img_array[:,:,c].min()
                max_val = img_array[:,:,c].max()
                result[:,:,c] = np.interp(modified[:,:,c], 
                                        (modified[:,:,c].min(), modified[:,:,c].max()), 
                                        (min_val, max_val))
        else:
            # For grayscale images
            min_val = img_array.min()
            max_val = img_array.max()
            result = np.interp(modified, (modified.min(), modified.max()), (min_val, max_val))
        
        # Ensure proper range and type
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to PIL, handling both RGB and grayscale
        if len(result.shape) == 2:
            result = np.repeat(result[..., None], 3, axis=2)
        return PILImage.fromarray(result)

class ResizeImages:
    """
    Transform to resize images that weren't cropped (cropping happens before sampling).
    Resizes maintaining aspect ratio so max dimension is target_size, then pads to square.
    
    Handles both:
    - Larger images (under threshold): resize max dimension to target_size, pad to square
    - Smaller images: resize max dimension to target_size, pad to square
    
    This transform only handles resizing - cropping is done before sampling in vos_dataset.py.
    
    Args:
        target_size: Target size (default: 512)
    """
    def __init__(self, target_size=512):
        self.target_size = target_size

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # Get image size from first frame (all frames should be same size)
        first_frame = datapoint.frames[0]
        w, h = first_frame.data.size
        
        # If already target_size x target_size, no need to resize
        if h == self.target_size and w == self.target_size:
            return datapoint
        
        # Resize maintaining aspect ratio and pad to square
        return self._resize_and_pad_datapoint(datapoint)
    
    def _resize_and_pad_datapoint(self, datapoint: VideoDatapoint):
        """Resize so max dimension is target_size (maintaining aspect ratio), then pad to square"""
        for i, frame in enumerate(datapoint.frames):
            # Get actual image size (PIL Image: (width, height))
            w, h = frame.data.size
            max_dim = max(h, w)
            
            # Resize the image while maintaining aspect ratio so max dimension is target_size
            if max_dim != self.target_size:
                # Calculate new size ensuring max dimension is exactly target_size
                if h > w:
                    # Height is larger, make height = target_size
                    new_h = self.target_size
                    new_w = int(round(w * self.target_size / h))
                else:
                    # Width is larger or equal, make width = target_size
                    new_w = self.target_size
                    new_h = int(round(h * self.target_size / w))
                
                # Resize using the calculated dimensions (resize expects (w, h) tuple, reverses to (h, w) for F.resize)
                datapoint = resize(datapoint, i, (new_w, new_h), square=False)
                # Get actual size after resize (PIL Image: (width, height))
                w, h = datapoint.frames[i].data.size
                # Verify max dimension is exactly target_size
                max_dim_after = max(h, w)
                if max_dim_after != self.target_size:
                    raise ValueError(
                        f"Resize failed: expected max_dim={self.target_size}, got {max_dim_after} "
                        f"(w={w}, h={h}, original w={frame.data.size[0]}, h={frame.data.size[1]})"
                    )
            else:
                # Already target_size, but may need padding if not square
                w, h = frame.data.size

            # Pad the image to make it square (target_size x target_size)
            # Note: frame.data.size is (width, height)
            pad_h = max(0, self.target_size - h)
            pad_w = max(0, self.target_size - w)
            
            # Calculate padding on each side to center the image
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            # Pad on all sides to center the image
            padding = [pad_left, pad_top, pad_right, pad_bottom]  # left, top, right, bottom
            
            if pad_h > 0 or pad_w > 0:
                datapoint = pad(datapoint, i, padding)
                
            # Verify final size is exactly target_size x target_size
            # Note: PIL Image.size is (width, height)
            final_w, final_h = datapoint.frames[i].data.size
            if final_h != self.target_size or final_w != self.target_size:
                raise ValueError(
                    f"Frame {i} size mismatch after resize+pad: expected {self.target_size}x{self.target_size}, "
                    f"got {final_w}x{final_h} (w={final_w}, h={final_h}). "
                    f"Before pad: w={w}, h={h}, pad_w={pad_w}, pad_h={pad_h}"
                )
                
        
        return datapoint