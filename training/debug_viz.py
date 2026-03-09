# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Debug visualization for training predictions vs ground truth.
Generates comparison images showing predictions (top) vs ground truths (bottom).
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


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


def _load_font(size: int = 11):
    """Load a PIL font, falling back to the built-in default."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_text_outlined(draw, xy, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    """Draw text with a 1-pixel outline for contrast on any background."""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _draw_dashed_line(draw, x0, y0, x1, y1, color, dash=6, width=2):
    """Draw a dashed line using alternating filled/empty segments."""
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    steps = max(1, int(dist / dash))
    for i in range(0, steps, 2):
        ta = i / steps
        tb = min((i + 1) / steps, 1.0)
        draw.line(
            [(int(x0 + dx * ta), int(y0 + dy * ta)),
             (int(x0 + dx * tb), int(y0 + dy * tb))],
            fill=color, width=width,
        )


def _draw_circle(draw, cx, cy, r, fill, outline=(0, 0, 0), outline_width=2):
    draw.ellipse(
        [(cx - r, cy - r), (cx + r, cy + r)],
        fill=fill, outline=outline, width=outline_width,
    )


def create_temporal_matching_visualization(
    input,
    t0: int,
    t1: int,
    key_ids,
    key_centroids,
    query_ids,
    query_centroids,
    match_logits,
    match_targets,
    child_to_parent: dict,
    save_dir: str,
    step: int = 0,
) -> Optional[str]:
    """
    Create a PIL-based debug visualization for the temporal matching head.

    Layout
    ──────
    ┌────────────────── header bar (stats) ──────────────────┐
    │  frame t0  (keys)  │  gap  │  frame t1  (queries)      │
    └────────────────────────────────────────────────────────┘

    Lines cross the gap between the two panels:
      • Solid line   = predicted match.
      • Dashed line  = correct GT assignment when it differs from prediction.

    Query-circle colour coding
    ──────────────────────────
      Green  – correct direct-track prediction.
      Orange – correct division prediction (daughter → mother).
      Red    – wrong prediction.
      Grey   – GT is NULL (new / unmatched cell).

    Key circles are drawn with a cyan outline and coloured by cell ID.

    Args:
        input:            BatchedVideoDatapoint (needs img_batch).
        t0, t1:           Frame indices (key frame and query frame).
        key_ids:          [N_k] tensor of cell IDs at frame t0.
        key_centroids:    [N_k, 2] normalised (cx, cy) in [0, 1].
        query_ids:        [N_q] tensor of cell IDs at frame t1.
        query_centroids:  [N_q, 2] normalised (cx, cy) in [0, 1].
        match_logits:     [N_q, N_k + 1] raw matching logits.
        match_targets:    [N_q] ground-truth key indices (N_k = NO_MATCH).
        child_to_parent:  {daughter_id: parent_id} across the video.
        save_dir:         Directory to save the PNG.
        step:             Global training step (used for the filename).

    Returns:
        Path to the saved PNG, or None if saving failed.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Constants ────────────────────────────────────────────────────────────
    HEADER_H  = 36
    GAP       = 28
    CIRCLE_R  = 8
    LINE_W    = 2

    C_CORRECT_TRACK = (50,  210,  70)   # green
    C_CORRECT_DIV   = (255, 160,   0)   # orange
    C_WRONG         = (210,  50,  50)   # red
    C_NULL          = (150, 150, 150)   # grey  (GT = NULL)
    C_GT_DIFF       = (80,  130, 230)   # blue dashed  (correct GT ≠ pred)
    C_KEY_OUTLINE   = (0,   200, 220)   # cyan ring on key circles
    C_HEADER_BG     = (35,  35,  35)

    N_k = key_ids.shape[0]
    N_q = query_ids.shape[0]

    if N_k == 0 or N_q == 0:
        return None

    # ── Predicted / GT assignments ───────────────────────────────────────────
    pred_idx = match_logits.argmax(dim=1).cpu()   # [N_q]  values in 0..N_k
    gt_idx   = match_targets.cpu()                # [N_q]

    # ── Per-query labels (is this a division daughter?) ──────────────────────
    key_id_set  = {k.item() for k in key_ids}
    is_daughter = [
        q.item() in child_to_parent and child_to_parent[q.item()] in key_id_set
        for q in query_ids
    ]

    # ── Accuracy stats ───────────────────────────────────────────────────────
    correct    = (pred_idx == gt_idx).float()
    overall_acc = correct.mean().item()
    div_mask   = torch.tensor(is_daughter)
    div_acc    = correct[div_mask].mean().item() if div_mask.sum() > 0 else float("nan")

    # ── Denormalize images ───────────────────────────────────────────────────
    def _get_img_np(frame_idx):
        img  = input.img_batch[frame_idx, 0].detach().cpu().float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (img * std + mean).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    try:
        img0_np = _get_img_np(t0)
        img1_np = _get_img_np(t1)
    except Exception:
        return None

    H, W = img0_np.shape[:2]

    # ── Build combined canvas ─────────────────────────────────────────────────
    canvas_w = W * 2 + GAP
    canvas_h = H + HEADER_H
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:HEADER_H, :] = C_HEADER_BG
    canvas[HEADER_H:, :W]       = img0_np
    canvas[HEADER_H:, W + GAP:] = img1_np

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _load_font(11)

    # ── Coordinate helpers ───────────────────────────────────────────────────
    kc = key_centroids.detach().cpu()    # [N_k, 2]
    qc = query_centroids.detach().cpu()  # [N_q, 2]

    def _kpix(i):
        """Key centroid → canvas (x, y)."""
        return int(kc[i, 0] * W), int(kc[i, 1] * H) + HEADER_H

    def _qpix(i):
        """Query centroid → canvas (x, y)."""
        return int(W + GAP + qc[i, 0] * W), int(qc[i, 1] * H) + HEADER_H

    def _query_color(q):
        p, g, div = pred_idx[q].item(), gt_idx[q].item(), is_daughter[q]
        if g == N_k:
            return C_NULL
        if p == g:
            return C_CORRECT_DIV if div else C_CORRECT_TRACK
        return C_WRONG

    # ── Draw lines FIRST (underneath circles) ────────────────────────────────
    for q in range(N_q):
        qx, qy = _qpix(q)
        p, g   = pred_idx[q].item(), gt_idx[q].item()
        color  = _query_color(q)

        if p < N_k:                                # solid predicted line
            kx, ky = _kpix(p)
            draw.line([(kx, ky), (qx, qy)], fill=color, width=LINE_W)

        if g < N_k and g != p:                     # dashed GT line (when wrong)
            kx_gt, ky_gt = _kpix(g)
            _draw_dashed_line(draw, kx_gt, ky_gt, qx, qy, C_GT_DIFF, dash=5, width=LINE_W)

    # ── Draw key circles (left panel) ────────────────────────────────────────
    for ki in range(N_k):
        kx, ky = _kpix(ki)
        fill   = get_color_for_cell_id(key_ids[ki].item())
        _draw_circle(draw, kx, ky, CIRCLE_R, fill, outline=C_KEY_OUTLINE, outline_width=2)
        lbl  = str(key_ids[ki].item())
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        _draw_text_outlined(draw, (kx - tw // 2, ky - th // 2), lbl, font)

    # ── Draw query circles (right panel) ─────────────────────────────────────
    for qi in range(N_q):
        qx, qy = _qpix(qi)
        color  = _query_color(qi)
        _draw_circle(draw, qx, qy, CIRCLE_R, color)
        lbl  = str(query_ids[qi].item())
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        _draw_text_outlined(draw, (qx - tw // 2, qy - th // 2), lbl, font)

    # ── Panel labels ─────────────────────────────────────────────────────────
    _draw_text_outlined(draw, (6,  2), f"Frame {t0}  (keys)",    font, fill=(200, 200, 200))
    _draw_text_outlined(draw, (W + GAP + 6, 2), f"Frame {t1}  (queries)", font, fill=(200, 200, 200))

    # ── Stats header ─────────────────────────────────────────────────────────
    div_str = f"{div_acc:.0%}" if div_acc == div_acc else "n/a"   # nan guard
    stats   = (
        f"step {step}  |  acc {overall_acc:.0%}  |  div acc {div_str}  |"
        f"  Nk={N_k}  Nq={N_q}  Ndiv={sum(is_daughter)}"
    )
    _draw_text_outlined(draw, (canvas_w // 2 - 200, 2), stats, font, fill=(220, 220, 220))

    # ── Colour legend (right side of header) ─────────────────────────────────
    legend = [
        (C_CORRECT_TRACK, "track"),
        (C_CORRECT_DIV,   "div"),
        (C_WRONG,         "wrong"),
        (C_NULL,          "null"),
        (C_GT_DIFF,       "GT(dashed)"),
    ]
    lx = canvas_w - len(legend) * 80 - 10
    for lc, ll in legend:
        draw.rectangle([(lx, 10), (lx + 12, 22)], fill=lc, outline=(0, 0, 0))
        _draw_text_outlined(draw, (lx + 15, 10), ll, font, fill=(200, 200, 200))
        lx += 80

    save_path = os.path.join(save_dir, f"temporal_match_idx{step:06d}_t{t0}_t{t1}.png")
    img.save(save_path)
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

