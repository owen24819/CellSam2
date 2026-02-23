# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Debug visualization for training predictions vs ground truth.
Generates comparison images showing predictions (top) vs ground truths (bottom).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_color_for_cell_id(cell_id: int) -> tuple:
    """Generate a consistent color for a given cell ID."""
    # Use cell_id to generate a deterministic color
    hue = (cell_id * 0.618033988749895) % 1.0  # Golden ratio for better distribution
    # Convert HSV to RGB with high saturation and value
    h = hue * 6.0
    c = 1.0
    x = 1.0 - abs(h % 2 - 1)
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int(r * 255), int(g * 255), int(b * 255))


def get_largest_blob(mask: torch.Tensor) -> torch.Tensor:
    """
    Keep only the largest connected component in a binary mask.
    
    Args:
        mask: Binary mask tensor [H, W]
    
    Returns:
        Mask with only the largest blob
    """
    if mask.sum() == 0:
        return mask
    
    # Simple connected component analysis using morphological approach
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    
    # Use scipy for connected components if available, otherwise use simple approach
    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask_np)
        if num_features == 0:
            return mask
        # Find the largest component
        sizes = ndimage.sum(mask_np, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        largest_mask = (labeled == largest_label).astype(np.uint8)
        return torch.from_numpy(largest_mask).to(mask.device).bool()
    except ImportError:
        # Fallback: just return the original mask
        return mask


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float = 0.5,
    cell_id: Optional[int] = None,
    obj_score: Optional[float] = None,
    obj_score_thresh: float = 0.0
) -> np.ndarray:
    """
    Overlay a colored mask on an image with optional cell ID label.
    
    Args:
        image: RGB image [H, W, 3]
        mask: Binary mask [H, W]
        color: RGB color tuple
        alpha: Transparency (0=transparent, 1=opaque)
        cell_id: Optional cell ID to display on the mask
        obj_score: Optional object score logit for this mask
        obj_score_thresh: Threshold for object score (default 0.0)
    
    Returns:
        Image with mask overlay and label
    """
    result = image.copy()
    mask_bool = mask > 0
    for c in range(3):
        result[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            result[:, :, c]
        )
    
    # Add cell ID label if provided
    if cell_id is not None and mask_bool.any():
        # Find centroid of mask for label placement
        y_coords, x_coords = np.where(mask_bool)
        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))
        
        # Draw cell ID text (simple approach - draw white pixels for the number)
        # For better rendering, we'd use PIL or cv2, but let's keep it simple
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(result)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            
            # Draw text - color indicates if object score is above/below threshold
            # White text = below threshold, Black text = above threshold
            text = str(cell_id)
            bbox = draw.textbbox((centroid_x, centroid_y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = centroid_x - text_width // 2
            text_y = centroid_y - text_height // 2
            
            # Determine text color based on object score
            assert obj_score is not None, "Object score must be provided for visualization"
            if obj_score >= obj_score_thresh:
                # Above threshold: black text with white outline
                text_color = (0, 0, 0)
                outline_color = (255, 255, 255)
            else:
                # Below threshold: white text with black outline
                text_color = (255, 255, 255)
                outline_color = (0, 0, 0)
            
            # Draw outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), text, fill=outline_color, font=font)
            # Draw main text
            draw.text((text_x, text_y), text, fill=text_color, font=font)
            result = np.array(pil_img)
        except ImportError:
            # If PIL not available, skip text rendering
            pass
    
    return result


def create_debug_visualization(
    batch,
    outputs: List[Dict],
    save_dir: str,
    sample_idx: int,
    dataset_idx: int,
) -> str:
    """
    Create a debug visualization comparing predictions vs ground truths.
    
    Args:
        batch: BatchedVideoDatapoint containing images and ground truth
        outputs: List of model output dicts (one per frame)
        save_dir: Directory to save visualization
        sample_idx: Sample index within dataset percentage
        dataset_idx: Index in the dataset
    
    Returns:
        Path to saved visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = batch.num_frames
    
    # Get images - shape [T, B, C, H, W]
    images = batch.img_batch
    _, _, C, H, W = images.shape
    
    # Assume batch size B=1 for visualization
    video_idx = 0
    
    # Prepare frames for stitching
    pred_frames = []
    gt_frames = []
    
    output_idx = 0  # Track which output corresponds to which frame
    
    for t in range(T):
        # Get image for this frame and convert to numpy [H, W, 3]
        img = images[t, video_idx].detach().cpu()
        
        # Denormalize image (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Create copies for pred and gt overlays
        pred_img = img.copy()
        gt_img = img.copy()
        
        # Skip frames with no inputs
        if batch.no_inputs[t]:
            pred_frames.append(pred_img)
            gt_frames.append(gt_img)
            continue
        
        # Get ground truth masks and cell IDs for this frame (only real ones)
        is_real_t = batch.is_real[t]
        is_real_masks_t = batch.is_real_masks[t]
        gt_masks_t = batch.masks[t][is_real_masks_t]  # [num_real_masks, H_mask, W_mask]
        
        # Build mask-to-ID mapping accounting for cell divisions
        # In collate_fn, masks are ordered as:
        # 1. Masks for non-dividing cells (in object order, skipping cells with 2 daughters)
        # 2. Daughter masks appended at the end (2 per dividing cell)
        cell_ids_raw = batch.metadata.unique_objects_identifier[t][is_real_t][:, 1].detach().cpu()
        cell_divides_t = batch.cell_divides[t][is_real_t].detach().cpu()
        daughter_ids_t = batch.daughter_ids[t][is_real_t].detach().cpu()
        
        gt_cell_ids = []
        daughter_ids_to_add = []
        for i in range(len(cell_ids_raw)):
            if cell_divides_t[i]:
                # This cell is dividing - no mask for mother, daughters added at end
                d_ids = daughter_ids_t[i]
                for d_id in d_ids:
                    if d_id > 0:
                        daughter_ids_to_add.append(int(d_id))
            else:
                # Non-dividing cell - mask corresponds to this cell ID
                gt_cell_ids.append(int(cell_ids_raw[i]))
        
        # Append daughter IDs (they come after non-dividing cell masks)
        gt_cell_ids.extend(daughter_ids_to_add)
        gt_cell_ids = np.array(gt_cell_ids)
        
        # Resize GT masks to image size if needed
        if gt_masks_t.shape[-2:] != (H, W) and gt_masks_t.numel() > 0:
            gt_masks_t = F.interpolate(
                gt_masks_t.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).bool()
        
        # Get predictions for this frame
        pred_obj_scores = None
        if output_idx < len(outputs):
            pred_masks_t = outputs[output_idx].get("pred_masks_high_res", None)
            
            if pred_masks_t is not None:
                # Get predicted cell IDs from model's tracking (accounts for divisions)
                pred_cell_ids = outputs[output_idx].get("tracking_object_ids", None)
                if pred_cell_ids is not None:
                    pred_cell_ids = pred_cell_ids.detach().cpu().numpy()
                else:
                    # Fallback: use gt_cell_ids if tracking_object_ids not available
                    pred_cell_ids = gt_cell_ids[:len(pred_masks_t)] if len(pred_masks_t) <= len(gt_cell_ids) else gt_cell_ids
                
                # Get object score logits (post-division)
                pred_obj_score_logits = outputs[output_idx].get("pred_object_score_logits", None)
                assert pred_obj_score_logits is not None, "pred_object_score_logits must be present in outputs for visualization"
                pred_obj_scores = pred_obj_score_logits.detach().cpu().numpy().flatten()
                
                # Resize predictions to image size if needed  
                if pred_masks_t.shape[-2:] != (H, W) and pred_masks_t.numel() > 0:
                    pred_masks_t = F.interpolate(
                        pred_masks_t.float(),
                        size=(H, W),
                        mode='nearest'
                    )
                
                # Apply sigmoid and threshold for predictions
                pred_masks_t = (pred_masks_t.sigmoid() > 0.5).squeeze(1)  # [num_cells, H, W]
                
                # Overlay predictions on image
                num_pred_cells = pred_masks_t.shape[0]
                for cell_idx in range(num_pred_cells):
                    mask = pred_masks_t[cell_idx].detach().cpu()
                    # Get largest blob only
                    mask = get_largest_blob(mask)
                    if mask.sum() > 0:
                        cell_id = int(pred_cell_ids[cell_idx]) if cell_idx < len(pred_cell_ids) else cell_idx
                        color = get_color_for_cell_id(cell_id)
                        obj_score = float(pred_obj_scores[cell_idx])
                        pred_img = overlay_mask_on_image(pred_img, mask.numpy(), color, alpha=0.6, cell_id=cell_id, obj_score=obj_score, obj_score_thresh=0.0)
            
            output_idx += 1
        
        # Overlay ground truths on image
        num_gt_cells = gt_masks_t.shape[0] if gt_masks_t.numel() > 0 else 0
        for cell_idx in range(num_gt_cells):
            mask = gt_masks_t[cell_idx].detach().cpu()
            # Get largest blob only
            mask = get_largest_blob(mask)
            if mask.sum() > 0:
                cell_id = int(gt_cell_ids[cell_idx]) if cell_idx < len(gt_cell_ids) else cell_idx
                color = get_color_for_cell_id(cell_id)
                # Ground truth always gets high score (10.0) to show black text
                gt_img = overlay_mask_on_image(gt_img, mask.numpy(), color, alpha=0.6, cell_id=cell_id, obj_score=10.0, obj_score_thresh=0.0)
        
        pred_frames.append(pred_img)
        gt_frames.append(gt_img)
    
    # Stitch frames horizontally: frame 0 to N from left to right
    # Top row: predictions, Bottom row: ground truths
    pred_row = np.concatenate(pred_frames, axis=1)  # [H, T*W, 3]
    gt_row = np.concatenate(gt_frames, axis=1)  # [H, T*W, 3]
    
    # Add labels
    label_height = 30
    pred_label = np.ones((label_height, pred_row.shape[1], 3), dtype=np.uint8) * 40
    gt_label = np.ones((label_height, gt_row.shape[1], 3), dtype=np.uint8) * 40
    
    # Stack vertically: label + pred row + separator + label + gt row
    separator = np.ones((5, pred_row.shape[1], 3), dtype=np.uint8) * 128
    final_image = np.concatenate([
        pred_label,
        pred_row,
        separator,
        gt_label,
        gt_row
    ], axis=0)
    
    # Add text labels using simple drawing
    # Mark frame indices at top
    for t in range(T):
        x_start = t * W + 5
        # Simple frame number indicator (draw a small number area)
        final_image[5:25, x_start:x_start+30] = [255, 255, 255]
    
    # Save the visualization
    save_path = os.path.join(save_dir, f"debug_sample_{sample_idx}_idx_{dataset_idx}.png")
    Image.fromarray(final_image).save(save_path)
    
    return save_path


def should_visualize(data_iter: int, total_iters: int, interval_percent: float = 5.0) -> bool:
    """
    Determine if we should create a visualization at this iteration.
    
    Args:
        data_iter: Current iteration index
        total_iters: Total number of iterations in epoch
        interval_percent: Percentage interval for visualization (default 5%)
    
    Returns:
        True if we should visualize at this iteration
    """

    if total_iters == 0:
        return False
    
    interval = max(1, int(total_iters * interval_percent / 100))
    return data_iter % interval == 0

