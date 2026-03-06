import math
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.amg import (
    MaskData,
    batched_mask_to_box,
)
from sam2.utils.misc import read_image
from sam2.utils.transforms import SAM2Transforms


class SAM2AutomaticCellTracker:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: int = 32,
        points_per_batch: int = 32,
        obj_score_thresh: float = 0.5,
        pred_iou_thresh: float = 0.7,
        div_obj_score_thresh: float = 0.5,
        box_nms_thresh: float = 0.5,
        max_hole_area: int = 0,
        max_sprinkle_area: int = 0,
        mask_threshold: float = 0.0,
        segment: bool = False,
        use_heatmap: bool = False,
        min_mask_area: int = 30,
        resize_threshold: Optional[int] = None,
        crop_overlap: int = 64,
        heatmap_debug: bool = False,
        heatmap_min_dist: int = 2,
        heatmap_threshold: float = 0.1,
        heatmap_topk: int = 0,
        segmentation_merge_iou_thresh: float = 0.5,
        crop_reassign_iou_thresh: float = 0.7,
        save_crop_movies: bool = False,
        postprocess_divisions: bool = False,
        aux_matching: Literal["off", "new_cells_only", "segment_then_aux_track"] = "off",
    ) -> None:
        """Initialize SAM2AutomaticCellTracker.
        
        Args:
            model: The SAM 2 model to use for mask prediction
            points_per_side: Number of points to sample along one side of the image
            points_per_batch: Number of points run simultaneously by the model
            obj_score_thresh: Object score threshold for filtering
            pred_iou_thresh: Predicted IoU threshold for filtering
            div_obj_score_thresh: Division object score threshold
            box_nms_thresh: Box IoU cutoff for non-maximal suppression
            max_hole_area: Maximum hole area to fill
            max_sprinkle_area: Maximum sprinkle area to remove
            mask_threshold: Threshold for binarizing mask logits
            segment: Whether to segment (True) or track (False)
            use_heatmap: Whether to use heatmap for point generation
            min_mask_area: Minimum mask area
            resize_threshold: Threshold for resizing
            crop_overlap: Overlap between crops
            heatmap_debug: Enable heatmap debugging
            heatmap_min_dist: Minimum distance for heatmap points
            heatmap_threshold: Heatmap threshold
            heatmap_topk: Top-k points from heatmap
            segmentation_merge_iou_thresh: IoU threshold for merging cells in segmentation
            crop_reassign_iou_thresh: IoU threshold for reassigning cells between crops
            postprocess_divisions: Enable post-processing to detect divisions across crops
            aux_matching: Temporal auxiliary matching mode: "off", "new_cells_only",
                or "segment_then_aux_track". Requires a model trained with the temporal
                matching head.
        """
        if aux_matching != "off":
            if not getattr(model, "enable_temporal_aux_matcher", False) or not hasattr(
                model, "temporal_matching_head"
            ) or model.temporal_matching_head is None:
                raise ValueError(
                    f"aux_matching={aux_matching!r} requires a model trained with the "
                    "temporal auxiliary matching head (enable_temporal_aux_matcher=True)."
                )
        self.model = model
        self.model.sam_mask_decoder.pred_iou_thresh = pred_iou_thresh
        self.model.sam_mask_decoder.obj_score_thresh = obj_score_thresh
        self.model.sam_mask_decoder.div_obj_score_thresh = div_obj_score_thresh
        self.device = model.device
        

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.obj_score_thresh = obj_score_thresh

        self.pred_iou_thresh = pred_iou_thresh
        self.div_obj_score_thresh = div_obj_score_thresh
        self.segment = segment
        self.use_heatmap = use_heatmap
        self.min_mask_area = min_mask_area
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.resize_threshold = resize_threshold
        self.crop_overlap = crop_overlap
        self.heatmap_debug = heatmap_debug
        self.heatmap_min_dist = heatmap_min_dist
        self.heatmap_threshold = heatmap_threshold
        self.heatmap_topk = heatmap_topk
        self.segmentation_merge_iou_thresh = segmentation_merge_iou_thresh
        self.crop_reassign_iou_thresh = crop_reassign_iou_thresh
        self.save_crop_movies = save_crop_movies
        self.postprocess_divisions = postprocess_divisions
        self.aux_matching = aux_matching

        if self.aux_matching == "segment_then_aux_track":
            self.segment = True

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

    def _list_frame_paths(self, video_path):
        video_path = Path(video_path)
        frame_paths = [
            p
            for p in video_path.iterdir()
            if p.suffix.lower()
            in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]
        ]
        frame_paths.sort(key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        if not frame_paths:
            raise RuntimeError(f"no images found in {video_path}")
        return frame_paths

    def _compute_crop_boxes(self, height, width, target_size, overlap):
        if height <= self.resize_threshold and width <= self.resize_threshold:
            return [(0, 0, width, height)]
        stride = max(1, target_size - overlap)
        
        # Compute number of crops needed per dimension
        if width <= target_size:
            xs = [0]
        else:
            num_cols = math.ceil((width - target_size) / stride) + 1
            # Evenly space crops so non-overlap grid cells align within crop bounds
            step_x = (width - target_size) / (num_cols - 1)
            xs = [round(i * step_x) for i in range(num_cols)]
        
        if height <= target_size:
            ys = [0]
        else:
            num_rows = math.ceil((height - target_size) / stride) + 1
            step_y = (height - target_size) / (num_rows - 1)
            ys = [round(i * step_y) for i in range(num_rows)]
        
        boxes = []
        for y0 in ys:
            for x0 in xs:
                x1 = min(width, x0 + target_size)
                y1 = min(height, y0 + target_size)
                boxes.append((x0, y0, x1, y1))
        return boxes

    def _crop_center(self, crop_box):
        x0, y0, x1, y1 = crop_box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def _compute_non_overlap_region(self, crop_box, crop_idx, all_crop_boxes, full_height, full_width):
        """Compute the non-overlapping region bounds for a crop.
        
        Divides image into grid based on number of crops. Each crop contributes one grid cell.
        Example: 100x100 image, 4 crops (2x2) -> stride_x=50, stride_y=50
        Non-overlap regions: (0,0,50,50), (50,0,100,50), (0,50,50,100), (50,50,100,100)
        
        Args:
            crop_box: (x0, y0, x1, y1) crop box in full image coordinates
            crop_idx: Index of this crop
            all_crop_boxes: List of all crop boxes
            full_height: Full image height
            full_width: Full image width
            
        Returns:
            (crop_offset_x, crop_offset_y, full_x0, full_y0, full_x1, full_y1)
        """
        if len(all_crop_boxes) == 1:
            # Single crop - use entire crop
            x0, y0, x1, y1 = crop_box
            return (0, 0, x0, y0, x1, y1)
        
        x0, y0, x1, y1 = crop_box
        
        # Find grid dimensions from crop positions
        # Get unique x0 and y0 positions to determine grid
        unique_x0 = sorted(set(box[0] for box in all_crop_boxes))
        unique_y0 = sorted(set(box[1] for box in all_crop_boxes))
        
        num_cols = len(unique_x0)
        num_rows = len(unique_y0)
        
        # Compute stride (non-overlap region size) from image dimensions
        stride_x = full_width // num_cols
        stride_y = full_height // num_rows
        
        # Find which grid cell this crop belongs to
        grid_x = unique_x0.index(x0)
        grid_y = unique_y0.index(y0)
        
        # Non-overlapping region is this grid cell
        full_x0 = grid_x * stride_x
        full_y0 = grid_y * stride_y
        full_x1 = min(full_width, full_x0 + stride_x)
        full_y1 = min(full_height, full_y0 + stride_y)
        
        # Offsets within crop mask
        offset_x = full_x0 - x0
        offset_y = full_y0 - y0
        
        return (offset_x, offset_y, full_x0, full_y0, full_x1, full_y1)

    def _assign_to_nearest_crop(self, centroids, crop_centers):
        assignments = []
        for cx, cy in centroids:
            dists = [
                (cx - center_x) ** 2 + (cy - center_y) ** 2
                for center_x, center_y in crop_centers
            ]
            assignments.append(int(np.argmin(dists)))
        return assignments

    def _assign_cells_to_crops(self, tiled_states, crop_centers, crop_masks, full_mask, frame_idx=0, divisions=None, relink_map=None):
        """Assign cells to crops for tracking continuity.
        
        Frame 0: Assign all cells to nearest crop (initial assignment)
        Frame 1+: Reassign based on centroid proximity + overlap confirmation
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_centers: List of (center_x, center_y) tuples for each crop
            crop_masks: List of crop masks (numpy arrays)
            full_mask: Already merged full mask
            frame_idx: Current frame index
            relink_map: Optional {seg_id: global_id} from aux matcher; used to set cell_id_map so local→global is correct.
            
        Returns:
            crop_assignments: List of sets, one per crop, containing global cell IDs
            cell_id_map: Dict mapping {global_id: local_id_in_assigned_crop}
                         Only contains entries where global_id != local_id (i.e. reassigned cells)
        """
        crop_assignments = [set() for _ in tiled_states]
        cell_id_map = {}  # {global_id: local_id_in_crop}
        if not tiled_states:
            return crop_assignments, cell_id_map

        # Extract cell IDs from merged mask (only cells that appear in current frame)
        obj_ids = np.unique(full_mask)
        obj_ids = obj_ids[obj_ids != 0]
        
        # Compute centroids for cells that appear in mask
        centroids = []
        obj_id_list = []
        obj_id_to_centroid = {}
        for obj_id in obj_ids:
            ys, xs = np.where(full_mask == obj_id)
            if len(xs) > 0:
                centroid = (xs.mean(), ys.mean())
                centroids.append(centroid)
                obj_id_list.append(int(obj_id))
                obj_id_to_centroid[int(obj_id)] = centroid

        # Frame 0: Assign all cells to nearest crop (global_id == local_id)
        if frame_idx == 0:
            assignments = self._assign_to_nearest_crop(centroids, crop_centers)
            for obj_id, assigned in zip(obj_id_list, assignments, strict=False):
                crop_assignments[assigned].add(obj_id)
            return crop_assignments, cell_id_map
        
        # Frame 1+: reassign based on centroid proximity + overlap
        # Use saved crop assignments and id map from previous frame
        cell_to_prev_crop = {}
        prev_cell_id_map = {}
        for crop_idx, state in enumerate(tiled_states):
            prev_assigned = state["crop_assignments"].get(frame_idx - 1, set())
            for obj_id in prev_assigned:
                cell_to_prev_crop[int(obj_id)] = crop_idx
            # Carry forward previous id map
            prev_map = state.get("cell_id_map", {}).get(frame_idx - 1, {})
            prev_cell_id_map.update(prev_map)
        
        # If a division occurred, assign daughter cells to the parent's previous crop
        if divisions:
            for parent_id, daughter_ids in divisions.items():
                parent_crop = cell_to_prev_crop.get(parent_id)
                if parent_crop is not None:
                    for d_id in daughter_ids:
                        if d_id not in cell_to_prev_crop:
                            cell_to_prev_crop[d_id] = parent_crop

        # Pre-populate cell_id_map and crop_assignments for cells linked by aux matcher (seg_id → global_id)
        relinked_global_ids = set()
        if relink_map:
            for seg_id, global_id in relink_map.items():
                global_id = int(global_id)
                relinked_global_ids.add(global_id)
                for crop_idx, cm in enumerate(crop_masks):
                    if seg_id in np.unique(cm):
                        cell_id_map[global_id] = int(seg_id)
                        crop_assignments[crop_idx].add(global_id)
                        break
        
        # For each cell that appears in the current mask
        for obj_id in obj_ids:
            obj_id_int = int(obj_id)
            if obj_id_int in relinked_global_ids:
                continue
            prev_crop = cell_to_prev_crop.get(obj_id_int)
            
            if prev_crop is None:
                # New cell (not tracked before) - assign to nearest crop
                centroid = obj_id_to_centroid[obj_id_int]
                assignments = self._assign_to_nearest_crop([centroid], crop_centers)
                assigned_crop = assignments[0]
                crop_assignments[assigned_crop].add(obj_id_int)
                # Find which local ID this crop uses for the cell
                crop_box = tiled_states[assigned_crop]["crop_box"]
                x0, y0, x1, y1 = crop_box
                cell_in_crop = (full_mask == obj_id_int)[y0:y1, x0:x1]
                if cell_in_crop.sum() > 0:
                    crop_mask = crop_masks[assigned_crop]
                    best_local_id = None
                    best_overlap = 0
                    for local_id in np.unique(crop_mask[cell_in_crop]):
                        if local_id == 0:
                            continue
                        overlap = np.logical_and(cell_in_crop, crop_mask == local_id).sum()
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_local_id = int(local_id)
                    if best_local_id is not None and best_local_id != obj_id_int:
                        cell_id_map[obj_id_int] = best_local_id
            else:
                centroid = obj_id_to_centroid.get(obj_id_int)
                if centroid is None:
                    crop_assignments[prev_crop].add(obj_id_int)
                    # Carry forward previous mapping if any
                    if obj_id_int in prev_cell_id_map:
                        cell_id_map[obj_id_int] = prev_cell_id_map[obj_id_int]
                        continue
                
                nearest_crop = self._assign_to_nearest_crop([centroid], crop_centers)[0]
                
                swap_assignments = False
                if nearest_crop != prev_crop:
                    # Check if cell overlaps with any cell in the nearest crop
                    current_cell_mask = (full_mask == obj_id_int)
                    crop_box = tiled_states[nearest_crop]["crop_box"]
                    x0, y0, x1, y1 = crop_box
                    crop_mask = crop_masks[nearest_crop]
                    
                    current_cell_in_crop = current_cell_mask[y0:y1, x0:x1]
                    cell_area = current_cell_in_crop.sum()
                    
                    if cell_area > 0:
                        # Find the best overlapping cell in the new crop
                        best_local_id = None
                        best_iou = 0
                        crop_cell_ids = np.unique(crop_mask)
                        crop_cell_ids = crop_cell_ids[crop_cell_ids != 0]
                        
                        for local_id in crop_cell_ids:
                            local_mask = (crop_mask == local_id)
                            intersection = np.logical_and(current_cell_in_crop, local_mask).sum()
                            union = np.logical_or(current_cell_in_crop, local_mask).sum()
                            iou = intersection / union if union > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_local_id = int(local_id)

                        # self.crop_reassign_iou_thresh should be high enough 
                        # so that when there is a discrency in timing of division            
                        # the cell is not reassigned and forced to wait until both crops have divided
                        if best_iou >= self.crop_reassign_iou_thresh and best_local_id is not None:
                            # Reassign to nearest crop with ID mapping
                            crop_assignments[nearest_crop].add(obj_id_int)
                            if best_local_id != obj_id_int:
                                cell_id_map[obj_id_int] = best_local_id
                            swap_assignments = True

                if not swap_assignments:
                    crop_assignments[prev_crop].add(obj_id_int)
                    if obj_id_int in prev_cell_id_map:
                        cell_id_map[obj_id_int] = prev_cell_id_map[obj_id_int]

        return crop_assignments, cell_id_map

    def _merge_crop_masks(self, tiled_states, crop_masks):
        """Merge crop masks using non-overlapping regions, then merge cells at boundaries.
        
        Step 1: Divide full_mask into non-overlapping regions (one per crop)
        Step 2: Add each crop to its respective non-overlapping area
        Step 3: For cells touching edges, check overlap with adjacent crops and combine
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_masks: List of crop masks (numpy arrays)
            
        Returns:
            Merged full_mask with cells combined across crop boundaries
        """
        full_height = tiled_states[0]["full_video_height"]
        full_width = tiled_states[0]["full_video_width"]
        full_mask = np.zeros((full_height, full_width), dtype=np.uint16)
        num_crops = len(tiled_states)
        
        if num_crops == 1:
            # Single crop - use entire crop
            state = tiled_states[0]
            crop_mask = crop_masks[0]
            x0, y0, x1, y1 = state["crop_box"]
            full_mask[y0:y1, x0:x1] = crop_mask
            return full_mask
        
        # Step 1 & 2: Divide into non-overlapping regions and add each crop to its area
        for state, crop_mask in zip(tiled_states, crop_masks, strict=False):
            crop_offset_x, crop_offset_y, full_x0, full_y0, full_x1, full_y1 = state["non_overlap_region"]
            
            # Extract non-overlapping region from crop mask
            actual_h = full_y1 - full_y0
            actual_w = full_x1 - full_x0
            crop_non_overlap = crop_mask[crop_offset_y:crop_offset_y + actual_h, 
                                        crop_offset_x:crop_offset_x + actual_w]
            
            # Place in full mask
            full_mask[full_y0:full_y1, full_x0:full_x1] = crop_non_overlap
        
        # Step 3: Find adjacent crops and merge cells at boundaries
        full_mask = self._merge_cells_at_boundaries(full_mask, tiled_states, crop_masks)
        
        # Step 4: Cleanup — fix any cells still badly clipped after boundary merge
        full_mask = self._cleanup_clipped_cells(full_mask, tiled_states, crop_masks)
        
        return full_mask

    def _filter_crop_masks_by_assignments(self, tiled_states, crop_masks, frame_idx):
        """Filter crop masks to only include assigned cells from previous frame and their daughters.
        
        For each crop, only keeps cells that were assigned to that crop in the previous frame,
        or their daughter cells (if the parent divided).
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_masks: List of crop masks to filter
            frame_idx: Current frame index
            
        Returns:
            filtered_masks: List of filtered crop masks
            prev_assignments: List of sets, one per crop, containing assigned cell IDs
        """
        prev_assignments = [
            state["crop_assignments"].get(frame_idx - 1, set())
            for state in tiled_states
        ]
        
        # Get previous cell_id_map (merged from all states)
        prev_id_map = {}
        for state in tiled_states:
            prev_map = state.get("cell_id_map", {}).get(frame_idx - 1, {})
            prev_id_map.update(prev_map)
        
        # Filter each crop: use assigned cells from previous frame, handle divisions
        filtered_masks = []
        for crop_idx, (state, track_mask) in enumerate(zip(tiled_states, crop_masks, strict=False)):
            filtered_mask = np.zeros_like(track_mask)
            assigned_cell_ids = prev_assignments[crop_idx]
            
            # Get parent_ids for this crop at current frame
            parent_map = {}
            if "obj_ids" in state and "parent_ids" in state:
                if frame_idx in state["obj_ids"] and frame_idx in state["parent_ids"]:
                    local_ids = state["obj_ids"][frame_idx].cpu().numpy()
                    local_parents = state["parent_ids"][frame_idx].cpu().numpy()
                    for obj_id, parent_id in zip(local_ids, local_parents, strict=False):
                        parent_map[int(obj_id)] = int(parent_id)
            
            # For each assigned cell, find it or its daughters
            for assigned_cell_id in assigned_cell_ids:
                # Look up the local cell ID in this crop (may differ if reassigned)
                local_cell_id = prev_id_map.get(assigned_cell_id, assigned_cell_id)
                
                # Check if local cell exists in current crop
                cell_mask = (track_mask == local_cell_id)
                if cell_mask.sum() > 0:
                    # Use the global ID in the filtered mask
                    filtered_mask[cell_mask] = assigned_cell_id
                else:
                    # Cell doesn't exist - check for daughters (cells with this as parent)
                    daughter_ids = [obj_id for obj_id, parent_id in parent_map.items() 
                                  if parent_id == local_cell_id]
                    for daughter_id in daughter_ids:
                        daughter_mask = (track_mask == daughter_id)
                        if daughter_mask.sum() > 0:
                            filtered_mask[daughter_mask] = daughter_id
            
            filtered_masks.append(filtered_mask)
        
        return filtered_masks, prev_assignments

    def _overlay_tracked_masks(self, tiled_states, crop_masks, prev_assignments=None):
        """Overlay full crop predictions for tracked cells.
        
        Unlike _merge_crop_masks which uses non-overlapping regions, this overlays
        the full prediction from each crop. Cells are already filtered to assigned ones.
        
        When overlaying, if a cell from a later crop overlaps significantly with a cell
        from an earlier crop, they are the same physical cell. We keep the ID from the
        crop that was assigned to track it, and drop the other ID.
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_masks: List of filtered crop masks (only assigned cells)
            prev_assignments: List of sets, one per crop, containing assigned cell IDs
            
        Returns:
            Merged full_mask with full cell predictions overlaid
        """
        full_height = tiled_states[0]["full_video_height"]
        full_width = tiled_states[0]["full_video_width"]
        full_mask = np.zeros((full_height, full_width), dtype=np.uint16)
        
        # Track which cells have been merged (old_id -> new_id mapping)
        # When two cells overlap, we keep the ID from the assigned crop
        cell_id_mapping = {}
        
        # Overlay each crop's full prediction
        for crop_idx, (state, crop_mask) in enumerate(zip(tiled_states, crop_masks, strict=False)):
            x0, y0, x1, y1 = state["crop_box"]
            crop_region = full_mask[y0:y1, x0:x1]
            crop_mask_nonzero = crop_mask > 0
            
            # Get assigned cells for this crop (to prefer their IDs)
            assigned_cells_this_crop = set()
            if prev_assignments is not None and crop_idx < len(prev_assignments):
                assigned_cells_this_crop = prev_assignments[crop_idx]
            
            # Check for overlaps with existing cells in this region
            existing_cells = np.unique(crop_region[crop_mask_nonzero])
            existing_cells = existing_cells[existing_cells != 0]
            
            # For each cell in current crop, check if it overlaps with existing cells
            crop_cell_ids = np.unique(crop_mask[crop_mask_nonzero])
            crop_cell_ids = crop_cell_ids[crop_cell_ids != 0]
            
            for crop_cell_id in crop_cell_ids:
                crop_cell_mask = (crop_mask == crop_cell_id)
                crop_cell_in_region = crop_cell_mask
                
                # Check overlap with existing cells in this region
                for existing_cell_id in existing_cells:
                    existing_cell_mask = (crop_region == existing_cell_id)
                    overlap = np.logical_and(crop_cell_in_region, existing_cell_mask)
                    overlap_area = overlap.sum()
                    
                    if overlap_area > 0:
                        # Check if significant overlap (same physical cell)
                        crop_cell_area = crop_cell_in_region.sum()
                        existing_cell_area = existing_cell_mask.sum()
                        overlap_ratio = overlap_area / min(crop_cell_area, existing_cell_area)
                        
                        if overlap_ratio >= 0.3:  # Significant overlap - same cell
                            # Determine which ID to keep based on assignments
                            # Prefer the ID from the crop that was assigned to track it
                            crop_cell_assigned = crop_cell_id in assigned_cells_this_crop
                            # Check if existing_cell_id was assigned to a previous crop
                            existing_cell_assigned = False
                            if prev_assignments is not None:
                                for prev_crop_idx, prev_assigned in enumerate(prev_assignments):
                                    if existing_cell_id in prev_assigned:
                                        existing_cell_assigned = True
                                        break
                            
                            # Keep the ID from the assigned crop, or current crop if both/neither assigned
                            if crop_cell_assigned and not existing_cell_assigned:
                                # Current crop's cell is assigned, keep it
                                cell_id_mapping[existing_cell_id] = crop_cell_id
                                # Update existing pixels
                                full_mask[y0:y1, x0:x1][existing_cell_mask] = crop_cell_id
                            elif existing_cell_assigned and not crop_cell_assigned:
                                # Existing cell is assigned, keep it (map current to existing)
                                cell_id_mapping[crop_cell_id] = existing_cell_id
                            else:
                                # Both or neither assigned - keep the one from later crop (current)
                                cell_id_mapping[existing_cell_id] = crop_cell_id
                                full_mask[y0:y1, x0:x1][existing_cell_mask] = crop_cell_id
            
            # Apply ID mappings to current crop mask before overlaying
            mapped_crop_mask = crop_mask.copy()
            for old_id, new_id in cell_id_mapping.items():
                mapped_crop_mask[crop_mask == old_id] = new_id
            
            # Overlay the mapped crop mask
            crop_region = full_mask[y0:y1, x0:x1]
            mapped_crop_mask_nonzero = mapped_crop_mask > 0
            crop_region[mapped_crop_mask_nonzero] = mapped_crop_mask[mapped_crop_mask_nonzero]
            full_mask[y0:y1, x0:x1] = crop_region
        
        return full_mask

    def _merge_cells_at_boundaries(self, full_mask, tiled_states, crop_masks):
        """Merge cells that overlap between adjacent crops.
        
        For each pair of adjacent crops, check IoU > 0.3 in their overlapping area.
        
        Args:
            full_mask: Initial merged mask with non-overlapping regions
            tiled_states: List of inference states, one per crop
            crop_masks: List of crop masks (numpy arrays)
            
        Returns:
            Merged full_mask with cells combined across crop boundaries
        """
        num_crops = len(tiled_states)
        
        if num_crops == 1:
            return full_mask
        
        # Check all pairs of crops
        for i in range(num_crops):
            state_i = tiled_states[i]
            crop_box_i = state_i["crop_box"]
            _, _, full_x0_i, full_y0_i, full_x1_i, full_y1_i = state_i["non_overlap_region"]
            
            for j in range(i + 1, num_crops):
                state_j = tiled_states[j]
                crop_box_j = state_j["crop_box"]
                _, _, full_x0_j, full_y0_j, full_x1_j, full_y1_j = state_j["non_overlap_region"]
                
                # Get overlapping area in crop boxes (not non-overlap regions)
                x0_i, y0_i, x1_i, y1_i = crop_box_i
                x0_j, y0_j, x1_j, y1_j = crop_box_j
                crop_overlap_x0 = max(x0_i, x0_j)
                crop_overlap_y0 = max(y0_i, y0_j)
                crop_overlap_x1 = min(x1_i, x1_j)
                crop_overlap_y1 = min(y1_i, y1_j)
                
                # Skip if crops don't overlap
                if crop_overlap_x0 >= crop_overlap_x1 or crop_overlap_y0 >= crop_overlap_y1:
                    continue
                
                # Convert to crop-local coordinates
                crop_i_overlap_x0 = crop_overlap_x0 - x0_i
                crop_i_overlap_y0 = crop_overlap_y0 - y0_i
                crop_i_overlap_x1 = crop_overlap_x1 - x0_i
                crop_i_overlap_y1 = crop_overlap_y1 - y0_i
                
                crop_j_overlap_x0 = crop_overlap_x0 - x0_j
                crop_j_overlap_y0 = crop_overlap_y0 - y0_j
                crop_j_overlap_x1 = crop_overlap_x1 - x0_j
                crop_j_overlap_y1 = crop_overlap_y1 - y0_j
                
                # Extract overlapping regions from original crop masks
                overlap_mask_i = crop_masks[i][crop_i_overlap_y0:crop_i_overlap_y1, crop_i_overlap_x0:crop_i_overlap_x1]
                overlap_mask_j = crop_masks[j][crop_j_overlap_y0:crop_j_overlap_y1, crop_j_overlap_x0:crop_j_overlap_x1]
                
                # Get edge cells: cells touching boundaries of non-overlap regions in full_mask
                edge_cells_i = set()
                edge_cells_j = set()
                
                # Check if crops share an edge or diagonal corner
                # Diagonal cases first (both x and y boundaries meet at a corner)
                if full_x1_i == full_x0_j and full_y1_i == full_y0_j:
                    # i is top-left, j is bottom-right
                    edge_mask_i = full_mask[full_y1_i-3:full_y1_i, full_x1_i-3:full_x1_i]
                    edge_mask_j = full_mask[full_y0_j:full_y0_j+3, full_x0_j:full_x0_j+3]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                elif full_x1_j == full_x0_i and full_y1_j == full_y0_i:
                    # i is bottom-right, j is top-left
                    edge_mask_i = full_mask[full_y0_i:full_y0_i+3, full_x0_i:full_x0_i+3]
                    edge_mask_j = full_mask[full_y1_j-3:full_y1_j, full_x1_j-3:full_x1_j]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                elif full_x1_j == full_x0_i and full_y1_i == full_y0_j:
                    # i is top-right, j is bottom-left
                    edge_mask_i = full_mask[full_y1_i-3:full_y1_i, full_x0_i:full_x0_i+3]
                    edge_mask_j = full_mask[full_y0_j:full_y0_j+3, full_x1_j-3:full_x1_j]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                elif full_x1_i == full_x0_j and full_y1_j == full_y0_i:
                    # i is bottom-left, j is top-right
                    edge_mask_i = full_mask[full_y0_i:full_y0_i+3, full_x1_i-3:full_x1_i]
                    edge_mask_j = full_mask[full_y1_j-3:full_y1_j, full_x0_j:full_x0_j+3]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                # Straight edge cases
                elif full_x1_i == full_x0_j:  # i's right edge touches j's left edge
                    # Get cells at right edge of crop i and left edge of crop j
                    edge_mask_i = full_mask[full_y0_i:full_y1_i, full_x1_i-1:full_x1_i]
                    edge_mask_j = full_mask[full_y0_j:full_y1_j, full_x0_j:full_x0_j+1]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                elif full_y1_i == full_y0_j:  # i's bottom edge touches j's top edge
                    # Get cells at bottom edge of crop i and top edge of crop j
                    edge_mask_i = full_mask[full_y1_i-1:full_y1_i, full_x0_i:full_x1_i]
                    edge_mask_j = full_mask[full_y0_j:full_y0_j+1, full_x0_j:full_x1_j]
                    edge_cells_i.update(np.unique(edge_mask_i[edge_mask_i != 0]))
                    edge_cells_j.update(np.unique(edge_mask_j[edge_mask_j != 0]))
                else:
                    continue  # Crops don't share an edge
                
                # Filter to only cells that are on edges AND exist in overlap region of original crop masks
                overlap_cell_ids_i = np.unique(overlap_mask_i)
                overlap_cell_ids_i = overlap_cell_ids_i[overlap_cell_ids_i != 0]
                overlap_cell_ids_j = np.unique(overlap_mask_j)
                overlap_cell_ids_j = overlap_cell_ids_j[overlap_cell_ids_j != 0]
                
                cell_ids_i = [c for c in edge_cells_i if c in overlap_cell_ids_i]
                cell_ids_j = [c for c in edge_cells_j if c in overlap_cell_ids_j]
                
                # Check all pairs of cells in the overlap
                # Only merge if they have the SAME cell ID (same physical cell detected in both crops)
                for cell_id_i in cell_ids_i:
                    cell_in_overlap_i = (overlap_mask_i == cell_id_i)
                    for cell_id_j in cell_ids_j:
                        cell_in_overlap_j = (overlap_mask_j == cell_id_j)
                        intersection = np.logical_and(cell_in_overlap_i, cell_in_overlap_j).sum()
                        union = np.logical_or(cell_in_overlap_i, cell_in_overlap_j).sum()
                        if union == 0:
                            continue
                        iou = intersection / union
                        
                        # If IoU > 0.3, merge them in full_mask (keep the one closer to its crop center)
                        if iou > 0.3:
                            # Get combined mask of both cells
                            full_cell_mask = np.logical_or(full_mask == cell_id_i, full_mask == cell_id_j)
                            
                            # Compute centroid of combined cell
                            cy, cx = np.where(full_cell_mask)
                            if len(cx) > 0:
                                centroid = (cx.mean(), cy.mean())
                                
                                # Get crop centers for these two crops
                                crop_centers_pair = [
                                    self._crop_center(state_i["crop_box"]),
                                    self._crop_center(state_j["crop_box"])
                                ]
                                
                                # Assign to nearest crop (i or j)
                                assignments = self._assign_to_nearest_crop([centroid], crop_centers_pair)
                                
                                # Use the cell from the assigned crop
                                if assignments[0] == 0:  # Closer to crop i
                                    full_mask[full_cell_mask] = cell_id_i
                                else:  # Closer to crop j
                                    full_mask[full_cell_mask] = cell_id_j
        
        return full_mask

    def _cleanup_clipped_cells(self, full_mask, tiled_states, crop_masks):
        """Cleanup pass: fix cells still badly clipped after boundary merging.
        
        For each cell in full_mask, compare to its original crop prediction.
        If IoU < 0.3, merge with an overlapping cell or remove.
        """
        cell_ids = np.unique(full_mask)
        cell_ids = cell_ids[cell_ids != 0]
        
        for cell_id in cell_ids:
            cell_mask = (full_mask == cell_id)
            if cell_mask.sum() == 0:
                continue
            
            # Find source crop with best overlap
            best_iou = 1.0
            best_crop_cell = None
            best_crop_box = None
            for state, crop_mask in zip(tiled_states, crop_masks, strict=False):
                crop_cell = (crop_mask == cell_id)
                if crop_cell.sum() == 0:
                    continue

                x0, y0, x1, y1 = state["crop_box"]
                cell_mask_crop = cell_mask[y0:y1, x0:x1]
                if cell_mask_crop.sum() == 0:
                    continue

                intersection = np.logical_and(cell_mask_crop, crop_cell).sum()
                union = np.logical_or(cell_mask_crop, crop_cell).sum()
                iou = intersection / union if union > 0 else 0
                if intersection > 0 and iou < best_iou:
                    best_iou = iou
                    best_crop_cell = crop_cell
                    best_crop_box = (x0, y0, x1, y1)
            
            if best_crop_cell is None or best_iou >= 0.3:
                continue
            
            # Cell is badly clipped — merge with overlapping cell or remove
            x0, y0, x1, y1 = best_crop_box
            crop_region = full_mask[y0:y1, x0:x1]
            overlapping = crop_region[best_crop_cell]
            overlapping = overlapping[(overlapping != 0) & (overlapping != cell_id)]
            if len(overlapping) > 0:
                merge_target = int(np.bincount(overlapping).argmax())
                full_mask[cell_mask] = merge_target
            else:
                full_mask[cell_mask] = 0
        
        return full_mask

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks.
        
        Args:
            mask1: Binary mask (same shape as mask2)
            mask2: Binary mask (same shape as mask1)
            
        Returns:
            IoU score between 0 and 1
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _allocate_obj_ids(self, inference_state, count):
        if count == 0:
            return torch.zeros(0, device=self.device, dtype=torch.int32)
        if "global_id_state" in inference_state:
            # global_id_state is a shared dictionary reference across all crops
            # Modifying it here updates the shared state visible to all crops
            start = inference_state["global_id_state"]["value"]
            obj_ids = torch.arange(
                start + 1,
                start + count + 1,
                device=self.device,
                dtype=torch.int32,
            )
            inference_state["global_id_state"]["value"] = start + count
            inference_state["max_obj_id"] = inference_state["global_id_state"]["value"]
            return obj_ids
        start = inference_state.get("max_obj_id", 0)
        obj_ids = torch.arange(
            start + 1, start + count + 1, device=self.device, dtype=torch.int32
        )
        inference_state["max_obj_id"] = start + count
        return obj_ids

    def _load_cropped_frame_tensor(self, inference_state, frame_idx):
        frame_path = inference_state["frame_paths"][frame_idx]
        image = read_image(frame_path)
        x0, y0, x1, y1 = inference_state["crop_box"]
        image = image.crop((x0, y0, x1, y1))
        transforms = inference_state["transforms"]
        image = transforms(image)
        if not inference_state["offload_video_to_cpu"]:
            image = image.to(inference_state["device"])
        return image

    @torch.inference_mode()
    def init_states(
        self,
        video_path,
        res_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        max_frame_num_to_track=None,
    ):
        def _build_state(
            *,
            resized_image_size,
            padding,
            images,
            num_frames,
            video_height,
            video_width,
            device,
            transforms,
            frame_paths=None,
            crop_box=None,
            full_video_height=None,
            full_video_width=None,
            global_id_state=None,
        ):
            inference_state = {
                "res_path": res_path,
                "video_path": video_path,
                "res_track": np.zeros((0, 4)),
                "resized_image_size": resized_image_size,
                "model_image_size": self.model.image_size,
                "padding": padding,
                "images": images,
                "num_frames": num_frames,
                "parent_ids": {},
                "max_frame_num_to_track": max_frame_num_to_track,
                "offload_video_to_cpu": offload_video_to_cpu,
                "offload_state_to_cpu": offload_state_to_cpu,
                "video_height": video_height,
                "video_width": video_width,
                "device": device,
                "storage_device": torch.device("cpu") if offload_state_to_cpu else device,
                "point_inputs": {},
                "cached_features": {},
                "constants": {},
                "obj_ids": None,
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
                "memory_dict": {"mask_mem_pos_enc": None},
                "transforms": transforms,
            }
            if frame_paths is not None:
                inference_state["frame_paths"] = frame_paths
            if crop_box is not None:
                inference_state["crop_box"] = crop_box
            if full_video_height is not None:
                inference_state["full_video_height"] = full_video_height
            if full_video_width is not None:
                inference_state["full_video_width"] = full_video_width
            if global_id_state is not None:
                inference_state["global_id_state"] = global_id_state
            # Store non-overlapping region bounds for merging
            if crop_box is not None and full_video_height is not None and full_video_width is not None:
                # Will be computed after we know num_crops
                inference_state["non_overlap_region"] = None
            return inference_state

        video_path = Path(video_path)
        compute_device = self.model.device

        assert video_path.is_dir(), f"Video path {video_path} is not a directory"

        frame_paths = self._list_frame_paths(video_path)
        first_frame = read_image(str(frame_paths[0]))
        full_width, full_height = first_frame.size
        target_size = self.model.image_size
        crop_boxes = self._compute_crop_boxes(
            full_height, full_width, target_size, self.crop_overlap
        )
        transforms = SAM2Transforms(
            resolution=target_size,
            mask_threshold=self.mask_threshold,
            max_hole_area=self.max_hole_area,
            max_sprinkle_area=self.max_sprinkle_area,
        )
        resized_image_size, padding = transforms._set_hw_params(
            first_frame.crop(crop_boxes[0]), target_size
        )
        transforms_list = [transforms] * len(crop_boxes)
        resized_paddings = [(resized_image_size, padding)] * len(crop_boxes)
        images_list = [[None] * len(frame_paths) for _ in crop_boxes]
        num_frames = len(frame_paths)

        crop_centers = [self._crop_center(box) for box in crop_boxes]
        global_id_state = {"value": 0}
        tiled_states = []
        for crop_idx, (crop_box, transforms, (resized_image_size, padding), images) in enumerate(zip(
            crop_boxes,
            transforms_list,
            resized_paddings,
            images_list,
            strict=False,
        )):
            inference_state = _build_state(
                resized_image_size=resized_image_size,
                padding=padding,
                images=images,
                num_frames=num_frames,
                video_height=crop_box[3] - crop_box[1],
                video_width=crop_box[2] - crop_box[0],
                device=compute_device,
                transforms=transforms,
                frame_paths=frame_paths,
                crop_box=crop_box,
                full_video_height=full_height,
                full_video_width=full_width,
                global_id_state=global_id_state,
            )
            # Compute and store non-overlapping region bounds
            inference_state["non_overlap_region"] = self._compute_non_overlap_region(
                crop_box, crop_idx, crop_boxes, full_height, full_width
            )
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
            tiled_states.append(inference_state)
        return tiled_states, crop_centers

    def predict(
        self,
        video_path,
        res_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        max_frame_num_to_track=None,
    ):
        """Predict and track cells throughout an image sequence directory.

        Args:
            video_path: Path to the image sequence directory
            offload_video_to_cpu: Whether to offload video frames to CPU to save GPU memory
            offload_state_to_cpu: Whether to offload inference state to CPU

        Returns:
            Dictionary of tracking results with frame indices as keys

        """
        tiled_states, crop_centers = self.init_states(
            video_path=video_path,
            res_path=res_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            max_frame_num_to_track=max_frame_num_to_track,
        )
        return self._predict_from_states(tiled_states, crop_centers, res_path, video_path)

    def _predict_from_states(self, tiled_states, crop_centers, res_path, video_path):

        generators = []
        for state in tiled_states:
            if not self.use_heatmap:
                state = self.generate_proportional_point_grid(state)
            generators.append(self.propagate_in_video(state, start_frame_idx=0, disable_progress=True))

        num_frames = tiled_states[0]["num_frames"] if tiled_states else 0
        tracking_results = []
        global_state = {
            "res_path": res_path,
            "video_path": video_path,
            "res_track": np.zeros((0, 4)),
            "video_height": tiled_states[0]["full_video_height"],
            "video_width": tiled_states[0]["full_video_width"],
            "obj_ids": {},
            "parent_ids": {},
        }
        if "frame_paths" in tiled_states[0]:
            global_state["frame_paths"] = tiled_states[0]["frame_paths"]

        # Initialize crop_assignments and cell_id_map storage in each state
        for state in tiled_states:
            state["crop_assignments"] = {}
            state["cell_id_map"] = {}
        
        # Track all global IDs ever used (to prevent reuse after cells disappear)
        all_used_global_ids = set()
        
        # Store crop tracking results if saving crop movies
        crop_tracking_results = [[] for _ in tiled_states] if self.save_crop_movies else None
        
        for frame_idx in tqdm(range(num_frames), desc="propagate in video"):
            aux_parent_map = {}
            aux_frame_divisions = {}
            relink_map = {}
            crop_masks = []
            for gen in generators:
                _, state, track_mask = next(gen)
                crop_masks.append(track_mask)

            # Store crop masks for movie saving
            if self.save_crop_movies:
                for crop_idx, crop_mask in enumerate(crop_masks):
                    crop_tracking_results[crop_idx].append(crop_mask)

            new_cell_candidates = []

            # Frame 0: Segmentation - merge all crop masks, then assign cells to crops
            if frame_idx == 0 or (self.segment and self.aux_matching == "off"):
                full_mask = self._merge_crop_masks(tiled_states, crop_masks)
            elif self.aux_matching == "segment_then_aux_track":
                # Per-crop temporal matching already applied in propagate_in_video.
                # Filter crop_masks to only assigned cells (and their daughters), then overlay.
                filtered_masks, prev_assignments = self._filter_crop_masks_by_assignments(
                    tiled_states, crop_masks, frame_idx
                )
                full_mask = self._overlay_tracked_masks(tiled_states, filtered_masks, prev_assignments)
            else:
                # Frame 1+: Tracking - use previous frame's assignments to filter current predictions
                filtered_masks, prev_assignments = self._filter_crop_masks_by_assignments(
                    tiled_states, crop_masks, frame_idx
                )
                # Merge tracked masks - overlay full predictions from each crop
                # Pass prev_assignments to prefer IDs from assigned crops
                tracked_mask = self._overlay_tracked_masks(tiled_states, filtered_masks, prev_assignments)
            
                # Get cells that appear in tracked_mask
                tracked_cell_ids = set(np.unique(tracked_mask))
                tracked_cell_ids.discard(0)
                
                # Merge all crops (like frame 0) to get segmentation mask for new cells
                seg_mask = self._merge_crop_masks(tiled_states, crop_masks)
                seg_cell_ids = np.unique(seg_mask)
                seg_cell_ids = seg_cell_ids[seg_cell_ids != 0]
                
                # Overlay: add new cells from seg_mask that don't overlap with tracked cells
                full_mask = tracked_mask.copy()

                for seg_cell_id in seg_cell_ids:
                    seg_cell_mask = (seg_mask == seg_cell_id)
                    # Check overlap with tracked cells
                    overlap_mask = np.logical_and(seg_cell_mask, tracked_mask > 0)
                    overlap_ratio = overlap_mask.sum() / seg_cell_mask.sum() if seg_cell_mask.sum() > 0 else 0
                    
                    if overlap_ratio >= 0.3:  # Significant overlap - merge into tracked cell
                        # Find which tracked cells this seg cell overlaps with
                        overlapping_cells = tracked_mask[overlap_mask]
                        if len(overlapping_cells) > 0:
                            cell_counts = np.bincount(overlapping_cells)
                            cell_counts[0] = 0  # ignore background
                            # Count how many tracked cells have significant overlap
                            seg_area = seg_cell_mask.sum()
                            significant_cells = []
                            for cid in range(len(cell_counts)):
                                if cell_counts[cid] > 0 and cell_counts[cid] / seg_area >= 0.1:
                                    significant_cells.append(cid)
                            
                            if len(significant_cells) == 1:
                                # Overlaps with exactly one tracked cell - safe to merge
                                tracked_cell_id = significant_cells[0]
                                gap_mask = np.logical_and(seg_cell_mask, full_mask == 0)
                                full_mask[gap_mask] = tracked_cell_id
                            # else: overlaps multiple tracked cells (e.g. division) - skip

                    else:
                        new_cell_mask = np.logical_and(seg_cell_mask, full_mask == 0)
                        if seg_cell_id in all_used_global_ids:
                            full_mask[new_cell_mask] = int(
                                self._allocate_obj_ids(tiled_states[0], 1)[0].item()
                            )
                        else:
                            full_mask[new_cell_mask] = seg_cell_id

                        new_cell_candidates.append(seg_cell_id)

            # Extract relink_map from per-crop temporal matching results for new cells
            if frame_idx > 0 and len(new_cell_candidates) > 0 and self.aux_matching == "new_cells_only":
                # Temporal matching already ran per-crop in propagate_in_video.
                temp_relink_map = {}
                # First, collect all potential matches
                for seg_cell_id in new_cell_candidates:
                    for state, crop_mask in zip(tiled_states, crop_masks, strict=False):
                        tm = state.get("temporal_matching_results", {}).get(frame_idx, {})
                        crop_relink_map = tm.get("relink_map", {})
                        if not crop_relink_map:
                            continue
                        x0, y0, x1, y1 = state["crop_box"]
                        crop_seg_region = seg_mask[y0:y1, x0:x1] if seg_mask is not None else None
                        if crop_seg_region is None:
                            continue
                        seg_cell_in_crop = (crop_seg_region == seg_cell_id)
                        if not np.any(seg_cell_in_crop):
                            continue
                        # Check if any local ID in relink_map overlaps with seg_cell_id region
                        for local_id, matched_id in crop_relink_map.items():
                            local_id_mask = (crop_mask == local_id)
                            overlap = np.logical_and(local_id_mask, seg_cell_in_crop)
                            if np.any(overlap):
                                # Check overlap ratio - must be > 0.7 to match
                                overlap_ratio = overlap.sum() / local_id_mask.sum() if local_id_mask.sum() > 0 else 0
                                if overlap_ratio > 0.7:
                                    # Map matched_id (local ID from prev frame) to global ID
                                    prev_l2g = state.get("local_to_global", {}).get(frame_idx - 1, {})
                                    
                                    # If matched_id is not in prev_l2g, check if it has a parent in current frame (just divided)
                                    if matched_id not in prev_l2g:
                                        # Check if matched_id has a parent in current frame
                                        current_parent_ids = state["parent_ids"][frame_idx]
                                        current_obj_ids = state.get("obj_ids", {}).get(frame_idx)
                                        if matched_id not in current_obj_ids:
                                            continue
                                        parent_id = current_parent_ids[current_obj_ids == matched_id][0].item()
                                        if parent_id == 0:
                                            continue
                                        global_matched_id = parent_id
                                    
                                    else:
                                        global_matched_id = prev_l2g.get(matched_id, matched_id)
                                    
                                    temp_relink_map[seg_cell_id] = global_matched_id
                                    break
                        if seg_cell_id in temp_relink_map:
                            break
                
                # Check which previous frame cells are already being tracked
                # tracked_mask contains cells that were successfully tracked, their IDs = previous frame IDs
                already_tracked_prev_ids = set(tracked_cell_ids) if 'tracked_cell_ids' in locals() else set()
                
                # Check for divisions: if multiple cells match to same previous frame cell
                matched_id_counts = {}
                for seg_id, matched_id in temp_relink_map.items():
                    matched_id_counts[matched_id] = matched_id_counts.get(matched_id, []) + [seg_id]
                
                # Process matches: handle divisions and skip cells that already divided
                for seg_cell_id, global_matched_id in temp_relink_map.items():
                    # Count how many cells are already tracking from this previous frame cell
                    already_tracked = 1 if global_matched_id in already_tracked_prev_ids else 0
                    new_matches = len(matched_id_counts.get(global_matched_id, []))
                    total_matches = already_tracked + new_matches
                    
                    # Skip if total matches would exceed division limit (2 daughters max)
                    if total_matches > 2:
                        continue
                    
                    # If this previous frame cell is already tracked, this is a division
                    if global_matched_id in already_tracked_prev_ids:
                        # Allocate new ID for the already-tracked cell that's becoming a daughter
                        new_daughter_id = int(self._allocate_obj_ids(tiled_states[0], 1)[0].item())
                        # Relabel the already-tracked cell in full_mask
                        full_mask[full_mask == global_matched_id] = new_daughter_id
                        # Update division set with new daughter ID
                        aux_frame_divisions.setdefault(global_matched_id, set()).add(seg_cell_id)
                        aux_frame_divisions[global_matched_id].add(new_daughter_id)
                        
                        # Add to aux_parent_map for later use in building parent_map
                        aux_parent_map[new_daughter_id] = global_matched_id
                        aux_parent_map[seg_cell_id] = global_matched_id
                        
                    # If multiple new cells match same parent, it's also a division
                    elif new_matches > 1:
                        aux_frame_divisions.setdefault(global_matched_id, set()).add(seg_cell_id)
                    else:
                        # Regular tracking (not a division) - relink to parent ID and apply to full_mask
                        relink_map[seg_cell_id] = global_matched_id
                        full_mask[full_mask == seg_cell_id] = global_matched_id

            # Ensure each cell is one blob (keep largest blob, remove smaller ones)
            for cell_id in np.unique(full_mask):
                if cell_id == 0:
                    continue
                cell_mask = (full_mask == cell_id)
                # Find connected components (blobs) for this cell
                num_labels, labels = cv2.connectedComponents(cell_mask.astype(np.uint8))
                if num_labels > 2:  # More than background (0) and one component (1)
                    # Find the largest blob
                    blob_sizes = []
                    for label in range(1, num_labels):  # Skip background (0)
                        blob_sizes.append((labels == label).sum())
                    largest_blob_idx = np.argmax(blob_sizes) + 1  # +1 because labels start at 1
                    # Remove all smaller blobs
                    for label in range(1, num_labels):
                        if label != largest_blob_idx:
                            full_mask[labels == label] = 0
            
            # Remove cells below area threshold
            for cell_id in np.unique(full_mask):
                if cell_id != 0 and (full_mask == cell_id).sum() < self.min_mask_area:
                    full_mask[full_mask == cell_id] = 0

            if not self.segment or self.aux_matching == "segment_then_aux_track":
                # Build local-to-global mapping for each crop (for post-processing divisions)
                for crop_idx, (state, crop_mask) in enumerate(zip(tiled_states, crop_masks, strict=False)):
                    x0, y0, x1, y1 = state["crop_box"]
                    crop_region = full_mask[y0:y1, x0:x1]
                    local_to_global = {}
                    for local_id in np.unique(crop_mask):
                        if local_id == 0:
                            continue
                        local_mask = (crop_mask == local_id)
                        overlapping = crop_region[local_mask]
                        overlapping = overlapping[overlapping != 0]
                        if len(overlapping) == 0:
                            continue
                        global_id = int(np.bincount(overlapping).argmax())
                        intersection = np.logical_and(local_mask, crop_region == global_id).sum()
                        union = np.logical_or(local_mask, crop_region == global_id).sum()
                        iou = intersection / union if union > 0 else 0
                        if iou > 0.3:
                            local_to_global[int(local_id)] = global_id
                    if "local_to_global" not in state:
                        state["local_to_global"] = {}
                    state["local_to_global"][frame_idx] = local_to_global

                # Build global divisions dict and parent_map for this frame using local_to_global mappings
                # Start with divisions from aux matcher (two new cells matched same parent)
                frame_divisions = {k: set(v) for k, v in aux_frame_divisions.items()}
                parent_map = dict(aux_parent_map)
                for crop_idx, state in enumerate(tiled_states):
                    if state["obj_ids"] is None or frame_idx not in state["obj_ids"]:
                        continue
                    if frame_idx not in state.get("parent_ids", {}):
                        continue
                    
                    # Get cells assigned to this crop in previous frame
                    prev_assignments = state.get("crop_assignments", {}).get(frame_idx - 1, set()) if frame_idx > 0 else set()
                    
                    l2g = state.get("local_to_global", {}).get(frame_idx, {})
                    prev_l2g = state.get("local_to_global", {}).get(frame_idx - 1, {}) if frame_idx > 0 else {}
                    local_ids = state["obj_ids"][frame_idx].cpu().numpy()
                    local_parents = state["parent_ids"][frame_idx].cpu().numpy()
                    
                    for obj_id, parent_id in zip(local_ids, local_parents, strict=False):
                        local_obj_int = int(obj_id)
                        local_parent_int = int(parent_id)
                        global_obj = l2g.get(local_obj_int, local_obj_int)
                        
                        # Build parent_map
                        if local_parent_int == 0:
                            parent_map.setdefault(global_obj, 0)
                        else:
                            # Convert local parent to global using current then previous frame mapping
                            global_par = l2g.get(local_parent_int)
                            if global_par is None:
                                global_par = prev_l2g.get(local_parent_int)
                            
                            # Only process if parent exists and is valid
                            if global_par is not None and global_par != global_obj:
                                # Only assign parent if it was assigned to this crop in previous frame (predicted division)
                                if frame_idx > 0 and global_par in prev_assignments:
                                    # Parent was assigned to this crop, so it predicted a division
                                    parent_map[global_obj] = global_par
                                    
                                    # Build frame_divisions: only add if parent was assigned to this crop
                                    # Use global_obj if it was mapped (not fallback), otherwise skip
                                    if local_obj_int in l2g:
                                        frame_divisions.setdefault(global_par, set()).add(global_obj)
                                # Otherwise, let postprocess_divisions handle it
                
                # Discard divisions that don't have exactly two daughter cells
                frame_divisions = {
                    parent: daughters 
                    for parent, daughters in frame_divisions.items() 
                    if len(daughters) == 2
                }

                # If a cell has a parent, remove any assignments where it's assigned as parent to other cells
                # (cells can't be both child and parent)
                for obj_id, parent_id in parent_map.items():
                    if parent_id in parent_map.keys():
                        parent_map[obj_id] = 0
            else:
                frame_divisions = None

            if frame_idx < num_frames:
                # Compute new assignments for current frame (to use in next frame)
                crop_assignments, cell_id_map = self._assign_cells_to_crops(
                    tiled_states, crop_centers, crop_masks, full_mask,
                    frame_idx=frame_idx, divisions=frame_divisions,
                    relink_map=relink_map,
                )
                # Store assignments and id map for next frame
                for crop_idx, state in enumerate(tiled_states):
                    state["crop_assignments"][frame_idx] = crop_assignments[crop_idx]
                    state["cell_id_map"][frame_idx] = cell_id_map
        
            tracking_results.append(full_mask)

            obj_ids = np.unique(full_mask)
            obj_ids = obj_ids[obj_ids != 0]
            all_used_global_ids.update(obj_ids.tolist())
            global_state["obj_ids"][frame_idx] = torch.tensor(
                obj_ids, device=self.device, dtype=torch.int32
            )
            parent_ids = np.zeros_like(obj_ids, dtype=np.int32)

            if not self.segment or self.aux_matching == "segment_then_aux_track":
                for i, obj_id in enumerate(obj_ids):
                    parent_ids[i] = parent_map.get(int(obj_id), 0)
            global_state["parent_ids"][frame_idx] = torch.tensor(
                parent_ids, device=self.device, dtype=torch.int32
            )

            self.save_ctc(full_mask, frame_idx, global_state)

        # Post-process divisions across crops
        if not self.segment and self.postprocess_divisions:
            self._postprocess_divisions(tiled_states, tracking_results, global_state)

        max_obj_id = 0
        for state in tiled_states:
            global_state_value = state.get("global_id_state", {}).get("value", 0)
            max_obj_id = max(max_obj_id, int(global_state_value))
        # Account for new daughter IDs created during post-processing
        if len(global_state["res_track"]) > 0:
            max_obj_id = max(max_obj_id, int(global_state["res_track"][:, 0].max()))
        global_state["max_obj_id"] = max_obj_id

        self.save_tracking_results(global_state, tracking_results)
        
        # Save individual crop movies if requested
        if self.save_crop_movies and crop_tracking_results:
            for crop_idx, crop_results in enumerate(crop_tracking_results):
                crop_box = tiled_states[crop_idx]["crop_box"]
                x0, y0, x1, y1 = crop_box
                crop_height = y1 - y0
                crop_width = x1 - x0
                
                
                crop_state = {
                    "res_path": res_path / 'crops',
                    "video_path": video_path,
                    "video_height": crop_height,
                    "video_width": crop_width,
                    "max_obj_id": global_state.get("max_obj_id", 0),
                    "frame_paths": tiled_states[crop_idx].get("frame_paths"),
                    "crop_box": crop_box,
                    "obj_ids": {},
                    "parent_ids": {},
                }
                # Extract obj_ids and parent_ids for each frame from crop state
                crop_state_obj = tiled_states[crop_idx]
                for frame_idx in range(len(crop_results)):
                    if "obj_ids" in crop_state_obj and crop_state_obj["obj_ids"] is not None:
                        if frame_idx in crop_state_obj["obj_ids"]:
                            crop_state["obj_ids"][frame_idx] = crop_state_obj["obj_ids"][frame_idx]
                    if "parent_ids" in crop_state_obj and crop_state_obj["parent_ids"] is not None:
                        if frame_idx in crop_state_obj["parent_ids"]:
                            crop_state["parent_ids"][frame_idx] = crop_state_obj["parent_ids"][frame_idx]
                
                # Build res_track for this crop so division lines appear in crop movies
                crop_res_track = np.zeros((0, 4))
                for frame_idx in range(len(crop_results)):
                    crop_mask = crop_results[frame_idx]
                    cell_ids = np.unique(crop_mask)
                    cell_ids = cell_ids[cell_ids != 0]
                    
                    # Build parent lookup for this frame from the crop's local IDs
                    parent_lookup = {}
                    if (crop_state_obj.get("obj_ids") is not None
                            and frame_idx in crop_state_obj["obj_ids"]
                            and crop_state_obj.get("parent_ids") is not None
                            and frame_idx in crop_state_obj["parent_ids"]):
                        local_ids = crop_state_obj["obj_ids"][frame_idx].cpu().numpy()
                        local_parents = crop_state_obj["parent_ids"][frame_idx].cpu().numpy()
                        for lid, lpar in zip(local_ids, local_parents, strict=False):
                            parent_lookup[int(lid)] = int(lpar)
                    
                    for cell_id in cell_ids:
                        cell_id_int = int(cell_id)
                        parent_id = parent_lookup.get(cell_id_int, 0)
                        
                        if len(crop_res_track) > 0 and cell_id_int in crop_res_track[:, 0].astype(int):
                            # Existing cell — extend its end frame
                            crop_res_track[crop_res_track[:, 0] == cell_id_int, 2] = frame_idx
                        else:
                            # New cell
                            crop_res_track = np.concatenate(
                                [crop_res_track, np.array([[cell_id_int, frame_idx, frame_idx, parent_id]])],
                                axis=0,
                            )
                
                crop_state["res_track"] = crop_res_track
                
                crop_state["res_path"].mkdir(parents=True, exist_ok=True)
                self.save_tracking_results(crop_state, crop_results, crop_idx=crop_idx)

                np.savetxt(crop_state["res_path"] / f"res_track_crop_{crop_idx}.txt", crop_res_track, fmt="%d")

        return tracking_results

    def generate_proportional_point_grid(self, inference_state):
        """Generate a grid of (x, y) points with density proportional to the image size.

        Args:
            inference_state: The video inference state

        Returns:
            points: torch.Tensor of shape [N, 2] where each row is (x, y)
            labels: torch.Tensor of shape [N] with all 1s (foreground)

        """
        resized_H, resized_W = inference_state["resized_image_size"]
        scale_factor_y = resized_H / self.model.image_size  # preserve relative density
        scale_factor_x = resized_W / self.model.image_size  # preserve relative density

        # Estimate number of points in each dimension
        points_y = int(self.points_per_side * scale_factor_y)
        points_x = int(self.points_per_side * scale_factor_x)

        # Avoid zero division or very few points
        points_y = max(1, points_y)
        points_x = max(1, points_x)

        # Generate evenly spaced coordinates with proper offsets
        # When only 1 point in a dimension, place it in the center
        if points_y == 1:
            ys = np.array([resized_H // 2], dtype=int)
        else:
            # Add offset to avoid placing points at the very edge
            offset_y = resized_H / (2 * points_y)
            ys = np.linspace(offset_y, resized_H - 1 - offset_y, points_y, dtype=int)

        if points_x == 1:
            xs = np.array([resized_W // 2], dtype=int)
        else:
            # Add offset to avoid placing points at the very edge
            offset_x = resized_W / (2 * points_x)
            xs = np.linspace(offset_x, resized_W - 1 - offset_x, points_x, dtype=int)

        # Images are center padded during training
        xs += (self.model.image_size - inference_state["resized_image_size"][1]) // 2
        ys += (self.model.image_size - inference_state["resized_image_size"][0]) // 2

        points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)[:, None]

        # Convert points to tensor
        points = torch.tensor(points, device=self.device, dtype=torch.float32)  # [N, 2]

        # Create corresponding labels tensor (all foreground)
        labels = torch.ones(
            (len(points), 1), dtype=torch.int, device=self.device
        )  # [N]

        # Save points and labels in inference_state for later reference
        inference_state["point_inputs"][0] = {
            "point_coords": points,
            "point_labels": labels,
        }

        if self.segment:
            for i in range(1, len(inference_state["images"])):
                inference_state["point_inputs"][i] = {
                    "point_coords": points,
                    "point_labels": labels,
                }

        return inference_state

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=0,
        disable_progress=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        num_frames = inference_state["num_frames"]
        max_frame_num_to_track = inference_state["max_frame_num_to_track"]

        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames

        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video", disable=disable_progress):
            if frame_idx == 0 or self.segment:
                if self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(
                        inference_state, frame_idx
                    )
                    inference_state["point_inputs"][frame_idx] = {
                        "point_coords": input_points,
                        "point_labels": point_labels,
                    }
                tracking_object_ids = None
                batch_size = inference_state["point_inputs"][frame_idx][
                    "point_coords"
                ].shape[0]
                is_init_cond_frame = True
            else:
                tracking_object_ids = inference_state["obj_ids"][frame_idx - 1]
                batch_size = len(tracking_object_ids)
                is_init_cond_frame = False

            if batch_size == 0:
                if inference_state["obj_ids"] is None:
                    inference_state["obj_ids"] = {}
                if "parent_ids" not in inference_state:
                    inference_state["parent_ids"] = {}
                inference_state["obj_ids"][frame_idx] = torch.zeros(
                    0, device=self.device, dtype=torch.int32
                )
                inference_state["parent_ids"][frame_idx] = torch.zeros(
                    0, device=self.device, dtype=torch.int32
                )

                if self.use_heatmap and not self.segment and frame_idx > 0:
                    input_points, point_labels = self.get_input_points_from_heatmap(
                        inference_state, frame_idx
                    )
                    inference_state["point_inputs"][frame_idx] = {
                        "point_coords": input_points,
                        "point_labels": point_labels,
                    }
                    batch_size = input_points.shape[0]
                    is_init_cond_frame = True
                    tracking_object_ids = None
                    if batch_size == 0:
                        track_mask = np.zeros(
                            (
                                inference_state["video_height"],
                                inference_state["video_width"],
                            ),
                            dtype=np.uint16,
                        )
                        yield frame_idx, inference_state, track_mask
                        continue
                else:
                    track_mask = np.zeros(
                        (
                            inference_state["video_height"],
                            inference_state["video_width"],
                        ),
                        dtype=np.uint16,
                    )
                    yield frame_idx, inference_state, track_mask
                    continue

            if batch_size > 0:
                # Retrieve image features (only need to compute once for all objects)
                (
                    _,
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._get_image_feature(inference_state, frame_idx, batch_size)

                # Run the core tracking step
                current_out, sam_outputs, high_res_features, _ = (
                    self.model._track_step(
                        is_init_cond_frame=is_init_cond_frame,
                        current_vision_feats=current_vision_feats,
                        current_vision_pos_embeds=current_vision_pos_embeds,
                        feat_sizes=feat_sizes,
                        point_inputs=inference_state["point_inputs"].get(
                            frame_idx, None
                        ),
                        mask_inputs=None,
                        num_frames=inference_state["num_frames"],
                        prev_sam_mask_logits=None,
                        tracking_object_ids=tracking_object_ids,
                        memory_dict=inference_state["memory_dict"],
                    )
                )

                # update cell tracks
                inference_state, track_mask = self.update_cell_tracks(
                    inference_state,
                    frame_idx,
                    sam_outputs,
                    current_out,
                    tracking_object_ids,
                )

                if not self.segment and frame_idx > 0 and self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(
                        inference_state, frame_idx
                    )

                    if input_points.shape[0] > 0:
                        input_points_copy = input_points.clone()

                        pad_left, pad_right, pad_top, pad_bottom = inference_state[
                            "padding"
                        ]
                        input_points[:, 0, 0] -= pad_left
                        input_points[:, 0, 1] -= pad_top

                        input_points[:, 0, 0] = input_points[:, 0, 0] * (
                            inference_state["video_width"]
                            / inference_state["resized_image_size"][1]
                        )
                        input_points[:, 0, 1] = input_points[:, 0, 1] * (
                            inference_state["video_height"]
                            / inference_state["resized_image_size"][0]
                        )

                        # Convert to numpy and int
                        input_points_np = input_points.cpu().numpy().astype(np.int32)
                        track_cell_ids = track_mask[
                            input_points_np[:, 0, 1], input_points_np[:, 0, 0]
                        ]

                        # Find indices where track_cell_ids is 0 (background)
                        background_point_indices = np.where(track_cell_ids == 0)[0]

                        if len(background_point_indices) > 0:
                            input_points = input_points_copy[background_point_indices]
                            point_labels = point_labels[background_point_indices]

                            inference_state["point_inputs"][frame_idx] = {
                                "point_coords": input_points,
                                "point_labels": point_labels,
                            }
                            batch_size = input_points.shape[0]

                            # Retrieve image features (only need to compute once for all objects)
                            (
                                _,
                                _,
                                current_vision_feats,
                                current_vision_pos_embeds,
                                feat_sizes,
                            ) = self._get_image_feature(
                                inference_state, frame_idx, batch_size
                            )

                            # Run the core tracking step
                            current_out, sam_outputs, high_res_features, _ = (
                                self.model._track_step(
                                    is_init_cond_frame=True,
                                    current_vision_feats=current_vision_feats,
                                    current_vision_pos_embeds=current_vision_pos_embeds,
                                    feat_sizes=feat_sizes,
                                    point_inputs=inference_state["point_inputs"].get(
                                        frame_idx, None
                                    ),
                                    mask_inputs=None,
                                    num_frames=inference_state["num_frames"],
                                    prev_sam_mask_logits=None,
                                    tracking_object_ids=None,
                                    memory_dict=inference_state["memory_dict"],
                                )
                            )

                            inference_state, detected_mask = self.update_cell_tracks(
                                inference_state,
                                frame_idx,
                                sam_outputs,
                                current_out,
                                heatmap_input=True,
                            )

                            if detected_mask.sum() > 0:
                                detected_cells = np.unique(detected_mask)
                                detected_cells = detected_cells[detected_cells != 0]

                                if len(inference_state["lost_obj_ids"][frame_idx]) > 0:
                                    lost_obj_ids = inference_state["lost_obj_ids"][
                                        frame_idx
                                    ]
                                    lost_high_res_masks = inference_state[
                                        "lost_high_res_masks"
                                    ][frame_idx]

                                    # Calculate IoU between each detected cell and lost cell
                                    ious = np.zeros(
                                        (len(detected_cells), len(lost_obj_ids))
                                    )
                                    for i, detected_id in enumerate(detected_cells):
                                        detected_mask_binary = (
                                            detected_mask == detected_id
                                        )
                                        for j, lost_id in enumerate(lost_obj_ids):
                                            intersection = np.logical_and(
                                                detected_mask_binary,
                                                lost_high_res_masks[j],
                                            ).sum()
                                            union = np.logical_or(
                                                detected_mask_binary,
                                                lost_high_res_masks[j],
                                            ).sum()
                                            ious[i, j] = (
                                                intersection / union if union > 0 else 0
                                            )

                                    # Find the lost cell with the highest IoU for each detected cell
                                    max_ious = np.max(ious, axis=1)
                                    lost_cell_indices = np.argmax(ious, axis=1)

                                    # Process each detected cell in order of IoU
                                    sorted_indices = np.argsort(
                                        -max_ious
                                    )  # Sort by descending IoU
                                    processed_lost_cells = set()
                                    cells_to_remove = (
                                        set()
                                    )  # Track which cells to remove

                                    for idx in sorted_indices:
                                        if (
                                            max_ious[idx] > 0
                                        ):  # If cell has positive object score, it is assumed to be match if there is any overlap
                                            detected_cell_id = int(detected_cells[idx])
                                            lost_idx = lost_cell_indices[idx]
                                            lost_cell_id = int(lost_obj_ids[lost_idx])

                                            # Skip if this lost cell was already matched
                                            if lost_cell_id in processed_lost_cells:
                                                continue

                                            if (
                                                frame_idx - 1
                                                in inference_state["memory_dict"][
                                                    lost_cell_id
                                                ]["frame_idx"]
                                            ):
                                                detected_mask[
                                                    detected_mask == detected_cell_id
                                                ] = lost_cell_id
                                                inference_state["memory_dict"][
                                                    lost_cell_id
                                                ]["mask_mem_features"] = torch.cat(
                                                    (
                                                        inference_state["memory_dict"][
                                                            lost_cell_id
                                                        ]["mask_mem_features"],
                                                        inference_state["memory_dict"][
                                                            detected_cell_id
                                                        ]["mask_mem_features"],
                                                    ),
                                                    dim=0,
                                                )
                                                inference_state["memory_dict"][
                                                    lost_cell_id
                                                ]["obj_ptr"] = torch.cat(
                                                    (
                                                        inference_state["memory_dict"][
                                                            lost_cell_id
                                                        ]["obj_ptr"],
                                                        inference_state["memory_dict"][
                                                            detected_cell_id
                                                        ]["obj_ptr"],
                                                    ),
                                                    dim=0,
                                                )
                                                inference_state["memory_dict"][
                                                    lost_cell_id
                                                ]["frame_idx"].append(frame_idx)
                                                inference_state["obj_ids"][frame_idx][
                                                    inference_state["obj_ids"][
                                                        frame_idx
                                                    ]
                                                    == detected_cell_id
                                                ] = lost_cell_id

                                                cells_to_remove.add(
                                                    detected_cell_id
                                                )  # Mark for removal
                                                processed_lost_cells.add(lost_cell_id)

                                                del inference_state["memory_dict"][
                                                    detected_cell_id
                                                ]

                                    # Remove the cells after processing all matches
                                    detected_cells = detected_cells[
                                        ~np.isin(detected_cells, list(cells_to_remove))
                                    ]

                                # Handle remaining detected cells
                                for detected_cell_id in detected_cells:
                                    # Get binary mask for current detected cell
                                    detected_mask_binary = (
                                        detected_mask == detected_cell_id
                                    )

                                    # Get all unique track IDs that overlap with this detected cell
                                    overlapping_track_ids = np.unique(
                                        track_mask[detected_mask_binary]
                                    )
                                    overlapping_track_ids = overlapping_track_ids[
                                        overlapping_track_ids > 0
                                    ]  # Remove background (0)

                                    if len(overlapping_track_ids) > 0:
                                        # Calculate IoU with each overlapping track
                                        best_iou = 0
                                        best_track_id = None

                                        for track_id in overlapping_track_ids:
                                            track_mask_binary = track_mask == track_id
                                            intersection = np.logical_and(
                                                detected_mask_binary, track_mask_binary
                                            ).sum()
                                            union = np.logical_or(
                                                detected_mask_binary, track_mask_binary
                                            ).sum()
                                            iou = (
                                                intersection / union if union > 0 else 0
                                            )

                                            if iou > best_iou:
                                                best_iou = iou
                                                best_track_id = track_id

                                        if (
                                            best_iou > 0.05
                                        ):  # If there's any overlap, assume it's the same cell
                                            # Update the detected mask to use the best matching track ID
                                            detected_mask[detected_mask_binary] = (
                                                best_track_id
                                            )
                                            inference_state["memory_dict"][
                                                best_track_id
                                            ]["mask_mem_features"][
                                                -1
                                            ] = inference_state["memory_dict"][
                                                detected_cell_id
                                            ]["mask_mem_features"][0]
                                            inference_state["memory_dict"][
                                                best_track_id
                                            ]["obj_ptr"][-1] = inference_state[
                                                "memory_dict"
                                            ][detected_cell_id]["obj_ptr"][0]
                                            inference_state["parent_ids"][frame_idx] = (
                                                inference_state["parent_ids"][
                                                    frame_idx
                                                ][
                                                    inference_state["obj_ids"][
                                                        frame_idx
                                                    ]
                                                    != detected_cell_id
                                                ]
                                            )
                                            inference_state["obj_ids"][frame_idx] = (
                                                inference_state["obj_ids"][frame_idx][
                                                    inference_state["obj_ids"][
                                                        frame_idx
                                                    ]
                                                    != detected_cell_id
                                                ]
                                            )

                                            del inference_state["memory_dict"][
                                                detected_cell_id
                                            ]

                                track_mask[(detected_mask > 0) * (track_mask == 0)] = (
                                    detected_mask[
                                        (detected_mask > 0) * (track_mask == 0)
                                    ]
                                )

                if self.aux_matching != "off":
                    track_mask = self._apply_temporal_matching_to_crop(
                        inference_state, frame_idx, track_mask
                    )

                yield frame_idx, inference_state, track_mask

    @torch.inference_mode()
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            if inference_state["images"][frame_idx] is None and "frame_paths" in inference_state:
                inference_state["images"][frame_idx] = self._load_cropped_frame_tensor(
                    inference_state, frame_idx
                )
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            # Clone the image to avoid inference mode tensor issues
            image = image.clone()
            backbone_out = self.model.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self.model._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def update_cell_tracks(
        self,
        inference_state,
        frame_idx,
        sam_outputs,
        current_out,
        tracking_object_ids=None,
        heatmap_input=False,
    ):
        """Update the cell tracks based on the current output and SAM outputs."""
        obj_ids = tracking_object_ids

        # Unpack SAM outputs
        (
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits_dict,
            div_score_logits,
            is_dividing,
        ) = sam_outputs

        #
        save_masks = torch.zeros_like(high_res_masks)

        # Keep only largest connected component for each mask
        for i in range(high_res_masks.shape[0]):
            mask = high_res_masks[i, 0].cpu().numpy()
            mask_binary = mask > self.mask_threshold
            if mask_binary.any():
                num_labels, labels = cv2.connectedComponents(
                    mask_binary.astype(np.uint8)
                )
                unique_labels, counts = np.unique(labels, return_counts=True)
                non_bg_mask = unique_labels != 0
                non_bg_labels = unique_labels[non_bg_mask]
                if non_bg_labels.size > 1:
                    largest_label = non_bg_labels[np.argmax(counts[non_bg_mask])]
                    mask_binary = labels == largest_label
                    
                save_masks[i, 0][
                    torch.from_numpy(mask_binary).to(self.device)
                ] = high_res_masks[i, 0][
                    torch.from_numpy(mask_binary).to(self.device)
                ]

                # Get max scores across all masks for each pixel
        argmax_scores = torch.max(save_masks[:, 0], dim=0)[1]  # shape: (H, W)
        # Count pixels for each mask index (excluding background)
        valid_mask = save_masks[:, 0].sum(0) > 0
        valid_indices = argmax_scores[valid_mask]
        max_mask_area = torch.bincount(
            valid_indices.flatten(), minlength=len(save_masks)
        )

        # Store ALL object scores BEFORE filtering for debugging
        # Use temporary indices if obj_ids not yet assigned
        if "all_output_dict_per_obj" not in inference_state:
            inference_state["all_output_dict_per_obj"] = {}
        if frame_idx not in inference_state["all_output_dict_per_obj"]:
            inference_state["all_output_dict_per_obj"][frame_idx] = {}
        
        # Store scores using obj_ids if available, otherwise use temporary indices
        num_objects = object_score_logits_dict["post_div"].shape[0]
        if obj_ids is not None and len(obj_ids) == num_objects:
            for i, obj_id in enumerate(obj_ids):
                inference_state["all_output_dict_per_obj"][frame_idx][obj_id.item()] = {
                    "pred_object_score_logits": object_score_logits_dict["post_div"][i, 0].item(),
                    "iou_pred": ious[i, 0].item(),
                    "mask_area": max_mask_area[i].item(),
                }
        else:
            # First frame or mismatch - use temporary indices
            for i in range(num_objects):
                temp_id = f"temp_{i}"
                inference_state["all_output_dict_per_obj"][frame_idx][temp_id] = {
                    "pred_object_score_logits": object_score_logits_dict["post_div"][i, 0].item(),
                    "iou_pred": ious[i, 0].item(),
                    "mask_area": max_mask_area[i].item(),
                }

        keep_tokens = (
            (object_score_logits_dict["post_div"][:, 0].sigmoid() > self.obj_score_thresh)
            * (ious[:, 0] > self.pred_iou_thresh)
            * (max_mask_area > self.min_mask_area)
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=high_res_masks[keep_tokens].flatten(0, 1),
            save_masks=save_masks[keep_tokens].flatten(0, 1),
            iou_preds=ious[keep_tokens].flatten(0, 1),
            obj_scores=object_score_logits_dict["post_div"][keep_tokens].flatten(0, 1),
            obj_ptr=obj_ptr[keep_tokens],
        )

        data["boxes"] = batched_mask_to_box(data["save_masks"] > self.mask_threshold)
        data["conf"] = data["obj_scores"].sigmoid() * data["iou_preds"]

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["conf"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        ).sort()[0]

        data.filter(keep_by_nms)

        removed_indices = torch.nonzero(keep_tokens)[
            ~torch.isin(
                torch.arange(keep_tokens.sum(), device=keep_tokens.device), keep_by_nms
            )
        ]
        keep_tokens[removed_indices] = False

        # Store which cells are predicted to be objects but are not kept by NMS or iou score or mask threshold
        valid_next_frame_mask = (
            object_score_logits_dict["post_div"][:, 0].sigmoid() > self.obj_score_thresh
        )

        if heatmap_input:
            obj_ids = self._allocate_obj_ids(inference_state, data["masks"].shape[0])
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"][frame_idx] = torch.cat(
                [inference_state["obj_ids"][frame_idx], obj_ids]
            )
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(
                len(obj_ids), device=self.device, dtype=torch.int32
            )
            inference_state["parent_ids"][frame_idx] = torch.cat(
                [inference_state["parent_ids"][frame_idx], parent_ids]
            )

        elif obj_ids is None:  # only in first frame
            num_cells = data["masks"].shape[0]
            obj_ids = self._allocate_obj_ids(inference_state, num_cells)
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"] = {frame_idx: obj_ids}
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(
                len(obj_ids), device=self.device, dtype=torch.int32
            )
            inference_state["parent_ids"] = {frame_idx: parent_ids}
            inference_state["lost_obj_ids"] = {
                frame_idx: torch.zeros(0, device=self.device, dtype=torch.int32)
            }
            inference_state["lost_high_res_masks"] = {}
        else:
            # Get all potential mother cells before NMS
            mother_ids = obj_ids[is_dividing]
            daughter_ids = self._allocate_obj_ids(inference_state, len(mother_ids) * 2)
            daughter_ids_list = daughter_ids.new_zeros(
                (len(obj_ids), 2), dtype=torch.int32
            )
            daughter_ids_list[is_dividing] = daughter_ids.reshape(-1, 2)

            prev_obj_ids = obj_ids.clone()

            # Update obj_ids to include all potential daughters, even if they might be removed by NMS
            obj_ids = torch.cat([obj_ids[~is_dividing], daughter_ids])

            # Now filter based on NMS results
            lost_obj_ids = obj_ids[valid_next_frame_mask * (~keep_tokens)]
            # If cell divided but is lost through NMS, we remove from error correction as this gets overly complicated
            lost_obj_ids = [obj_id for obj_id in lost_obj_ids if obj_id in prev_obj_ids]
            inference_state["lost_obj_ids"][frame_idx] = lost_obj_ids
            if len(lost_obj_ids) > 0:
                lost_high_res_masks = high_res_masks[
                    valid_next_frame_mask * (~keep_tokens)
                ].flatten(0, 1)
                lost_high_res_masks[
                    :, (data["masks"] > self.mask_threshold).sum(0) > 0
                ] = -torch.inf
                lost_high_res_masks = self.postprocess_mask(
                    lost_high_res_masks, inference_state
                )
                inference_state["lost_high_res_masks"][frame_idx] = (
                    lost_high_res_masks > self.mask_threshold
                )

            obj_ids = obj_ids[keep_tokens]

            parent_ids = torch.zeros(
                len(obj_ids), device=self.device, dtype=torch.int32
            )

            # Update parent IDs for daughters that survived NMS
            for mother_id, pair_daughter_ids in zip(
                mother_ids, daughter_ids.reshape(-1, 2), strict=False
            ):
                if pair_daughter_ids[0] in obj_ids and pair_daughter_ids[1] in obj_ids:
                    mask0 = obj_ids == pair_daughter_ids[0]
                    mask1 = obj_ids == pair_daughter_ids[1]
                    parent_ids[mask0] = mother_id
                    parent_ids[mask1] = mother_id
                else:
                    # if one of the daughter cells is not in the final_obj_ids due to nms, then the other daughter cell must be the mother cell
                    dau_id = (
                        pair_daughter_ids[0]
                        if pair_daughter_ids[0] in obj_ids
                        else pair_daughter_ids[1]
                    )
                    obj_ids[obj_ids == dau_id] = mother_id
                    mother_ids = mother_ids[mother_ids != mother_id]
                    daughter_ids_list = daughter_ids_list.clone()
                    daughter_ids_list[
                        torch.isin(daughter_ids_list, pair_daughter_ids)
                    ] = 0

            inference_state["obj_ids"][frame_idx] = obj_ids
            inference_state["parent_ids"][frame_idx] = parent_ids

        current_out["pred_masks_high_res"] = data["masks"]
        current_out["pred_object_score_logits"] = data["obj_scores"]
        current_out["obj_ptr"] = data["obj_ptr"]

        # Store object scores per FINAL object (after division handling) for debugging
        if "output_dict_per_obj" not in inference_state:
            inference_state["output_dict_per_obj"] = {}
        if frame_idx not in inference_state["output_dict_per_obj"]:
            inference_state["output_dict_per_obj"][frame_idx] = {}
        for i, obj_id in enumerate(obj_ids):
            inference_state["output_dict_per_obj"][frame_idx][obj_id.item()] = {
                "pred_object_score_logits": data["obj_scores"][i],
                "iou_pred": data["iou_preds"][i],
            }

        if not heatmap_input:
            assert current_out["pred_masks_high_res"].shape[0] == len(obj_ids)

        # Retrieve image features (only need to compute once for all objects)
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(
            inference_state, frame_idx, current_out["pred_masks_high_res"].shape[0]
        )

        # Build and store aux tokens for temporal matching (used as query tokens for current frame, key tokens for next frame)
        if (
            self.aux_matching != "off"
            and len(obj_ids) > 0
            and hasattr(self.model, "temporal_matching_head")
            and self.model.temporal_matching_head is not None
        ):
            # Use non-conditioned raw backbone features (reuse current_vision_feats)
            raw_feat = current_vision_feats[-1]
            H_feat, W_feat = feat_sizes[-1]
            C_feat = raw_feat.size(2)
            N = len(obj_ids)
            raw_pix_feat = raw_feat.permute(1, 2, 0).view(N, C_feat, H_feat, W_feat)
            
            # data["masks"] has shape [N, H, W] after flatten(0, 1), need [N, 1, H, W] for build_matching_tokens
            mask_logits = data["masks"].to(raw_pix_feat.device).unsqueeze(1)
            key_tokens, key_centroids, key_areas = (
                self.model.temporal_matching_head.build_matching_tokens(
                    data["obj_ptr"], raw_pix_feat, mask_logits
                )
            )
            if "aux_tokens_data" not in inference_state:
                inference_state["aux_tokens_data"] = {}
            inference_state["aux_tokens_data"][frame_idx] = {
                "tokens": key_tokens,
                "centroids": key_centroids,
                "areas": key_areas,
                "ids": obj_ids.cpu().numpy().tolist(),
            }

        if not self.segment and len(obj_ids) > 0:
            inference_state["memory_dict"] = self.model._update_memory_features(
                current_vision_feats,
                feat_sizes,
                inference_state["point_inputs"].get(frame_idx, None),
                run_mem_encoder=True,
                current_out=current_out,
                memory_dict=inference_state["memory_dict"],
                tracking_object_ids=obj_ids,
                frame_idx=frame_idx,
                mother_ids=mother_ids,
                prev_tracking_object_ids=prev_obj_ids,
                daughter_ids_list=daughter_ids_list,
            )

        assert data["save_masks"].shape[0] == data["masks"].shape[0]

        # If no masks are predicted, return an empty track mask
        if data["masks"].shape[0] == 0:
            track_mask = np.zeros(
                (inference_state["video_height"], inference_state["video_width"]),
                dtype=np.uint16,
            )
            return inference_state, track_mask

        track_mask = self.postprocess_mask(data["save_masks"], inference_state)

        # Get the maximum value and index across all masks at each pixel position
        max_values = np.max(track_mask, axis=0)  # returns max values
        arg_max = np.argmax(track_mask, axis=0)  # returns indices

        # Create a mask filled with zeros (background)
        track_mask = np.zeros_like(arg_max)

        # For pixels above threshold, assign the corresponding object ID
        valid_pixels = max_values > self.mask_threshold
        obj_ids_np = obj_ids.cpu().numpy()
        track_mask[valid_pixels] = obj_ids_np[arg_max[valid_pixels]]

        return inference_state, track_mask

    def postprocess_mask(self, masks, inference_state):
        pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]

        pad_right = inference_state["model_image_size"] - pad_right
        pad_bottom = inference_state["model_image_size"] - pad_bottom

        masks = masks[:, pad_top:pad_bottom, pad_left:pad_right]
        masks = masks.permute(1, 2, 0).cpu().numpy()
        masks = cv2.resize(
            masks, (inference_state["video_width"], inference_state["video_height"])
        )
        if masks.ndim == 2:
            masks = masks[None, ...]
        else:
            masks = masks.transpose(2, 0, 1)

        return masks

    def _apply_temporal_matching_to_crop(self, inference_state, frame_idx, track_mask):
        """Apply per-crop temporal matching: match current frame cells to prev frame cells.

        Runs the temporal matching head within a single crop. For "segment_then_aux_track",
        modifies track_mask so that matched cells inherit the previous frame's IDs. For
        "new_cells_only", preserves SAM2's track_mask and only stores relink_map for
        later use. Stores parent_map and divisions in inference_state for later global
        compilation.

        Returns:
            track_mask (modified for "segment_then_aux_track", unchanged for "new_cells_only").
        """
        parent_map = {}
        divisions = {}

        def _store_and_return():
            # For "new_cells_only", store in temporal_matching_results
            # For "segment_then_aux_track", store in parent_ids (done below)
            if self.aux_matching == "new_cells_only":
                inference_state.setdefault("temporal_matching_results", {})[frame_idx] = {
                    "relink_map": {},
                }
            return track_mask

        if frame_idx == 0:
            return _store_and_return()
        if (
            not hasattr(self.model, "temporal_matching_head")
            or self.model.temporal_matching_head is None
        ):
            return _store_and_return()

        # Query: current frame's cell tokens
        current_aux = inference_state.get("aux_tokens_data", {}).get(frame_idx)
        if current_aux is None or len(current_aux.get("ids", [])) == 0:
            return _store_and_return()

        # Key: previous frame's tokens
        prev_aux = inference_state.get("aux_tokens_data", {}).get(frame_idx - 1)
        if prev_aux is None or len(prev_aux.get("ids", [])) == 0:
            return _store_and_return()

        device = self.device
        query_tokens = current_aux["tokens"].to(device)
        query_centroids = current_aux["centroids"].to(device)
        query_areas = current_aux["areas"].to(device)
        query_ids = current_aux["ids"]

        key_tokens = prev_aux["tokens"].to(device)
        key_centroids = prev_aux["centroids"].to(device)
        key_areas = prev_aux["areas"].to(device)
        key_ids = prev_aux["ids"]

        # Run temporal matching head
        match_logits = self.model.temporal_matching_head.forward(
            query_tokens, key_tokens, query_centroids, key_centroids,
            query_areas, key_areas,
        )

        # Sparsify: keep only the max per query row
        best_col = match_logits.argmax(dim=1)
        sparse_mask = torch.nn.functional.one_hot(
            best_col, num_classes=match_logits.shape[1]
        ).to(match_logits.dtype)
        match_logits = match_logits * sparse_mask

        # Division-aware: each key (parent) can match up to 2 queries (daughters)
        relink_map = {}
        num_keys = len(key_ids)
        num_queries = match_logits.shape[0]
        for key_j in range(num_keys):
            key_id = key_ids[key_j]
            scores_j = match_logits[:, key_j]
            candidates = [
                (i, scores_j[i].item())
                for i in range(num_queries)
                if scores_j[i].item() > 0.0
            ]
            candidates.sort(key=lambda x: -x[1])
            top2 = candidates[:2]
            if not top2:
                continue
            if len(top2) == 1:
                (i, _) = top2[0]
                relink_map[query_ids[i]] = key_id
            else:
                (i0, _), (i1, _) = top2[0], top2[1]
                if self.aux_matching == "segment_then_aux_track":
                    # Create two new daughter IDs for divisions
                    daughter_ids_alloc = self._allocate_obj_ids(inference_state, 2)
                    new_id1 = int(daughter_ids_alloc[0].item())
                    new_id2 = int(daughter_ids_alloc[1].item())
                    relink_map[query_ids[i0]] = new_id1
                    relink_map[query_ids[i1]] = new_id2
                    parent_map[new_id1] = key_id
                    parent_map[new_id2] = key_id
                    divisions.setdefault(key_id, set()).add(new_id1)
                    divisions.setdefault(key_id, set()).add(new_id2)
                else:
                    # For "new_cells_only", just relink both to parent (no new IDs, no parent_map/divisions tracking)
                    relink_map[query_ids[i0]] = key_id
                    relink_map[query_ids[i1]] = key_id

        # Apply relink to track_mask (skip for "new_cells_only" to preserve SAM2 tracking)
        if self.aux_matching == "segment_then_aux_track":
            for old_id, new_id in relink_map.items():
                track_mask[track_mask == old_id] = new_id

            # Update aux_tokens_data with relinked IDs (skip for "new_cells_only" to preserve SAM2 IDs)
            aux_tokens = inference_state.get("aux_tokens_data", {}).get(frame_idx)
            if aux_tokens is not None:
                aux_tokens["ids"] = [relink_map.get(tid, tid) for tid in aux_tokens["ids"]]

            # For "segment_then_aux_track", update obj_ids and parent_ids to reflect relinking
            if  relink_map:
                obj_ids = inference_state.get("obj_ids", {}).get(frame_idx)
                if obj_ids is not None:
                    # Initialize parent_ids if needed
                    if "parent_ids" not in inference_state:
                        inference_state["parent_ids"] = {}
                    if frame_idx not in inference_state["parent_ids"]:
                        inference_state["parent_ids"][frame_idx] = torch.zeros(
                            len(obj_ids), device=self.device, dtype=torch.int32
                        )
                    
                    # Update obj_ids and parent_ids in one pass
                    obj_ids_np = obj_ids.cpu().numpy()
                    parent_ids = inference_state["parent_ids"][frame_idx].clone()
                    for i, obj_id in enumerate(obj_ids_np):
                        obj_id_int = int(obj_id)
                        # Update obj_id if relinked
                        if obj_id_int in relink_map:
                            obj_ids[i] = relink_map[obj_id_int]
                            obj_id_int = relink_map[obj_id_int]  # Use relinked ID for parent lookup
                        # Update parent_id if in parent_map
                        if obj_id_int in parent_map:
                            parent_ids[i] = parent_map[obj_id_int]
                    
                    inference_state["obj_ids"][frame_idx] = obj_ids
                    inference_state["parent_ids"][frame_idx] = parent_ids

        elif self.aux_matching == "new_cells_only":    
            inference_state.setdefault("temporal_matching_results", {})[frame_idx] = {
                "relink_map": relink_map,
            }

        return track_mask

    def get_input_points_from_heatmap(self, inference_state, frame_idx):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, 1)

        image_tensor = inference_state["images"][frame_idx]
        if image_tensor is not None:
            image_tensor = image_tensor.to(self.device)
        heatmap_predictions = self.model.get_heatmap_predictions(
            current_vision_feats, feat_sizes, image_tensor=image_tensor
        )[0, 0]
        input_points = self.model.extract_peak_points(
            heatmap_predictions,
            min_dist=self.heatmap_min_dist,
            threshold=self.heatmap_threshold,
        )
        if input_points.shape[0] == 0 and self.heatmap_topk > 0:
            input_points = self._extract_topk_points(heatmap_predictions, self.heatmap_topk)
        point_labels = torch.ones(
            (input_points.shape[0], 1), dtype=torch.int, device=self.device
        )

        if self.heatmap_debug and "frame_paths" in inference_state:
            self._save_heatmap_debug(
                inference_state,
                frame_idx,
                input_points,
                heatmap_predictions,
            )

        return input_points, point_labels

    def _extract_topk_points(self, heatmap, topk):
        heatmap = heatmap.sigmoid()
        flat = heatmap.flatten()
        k = min(topk, flat.shape[0])
        if k == 0:
            return torch.empty((0, 1, 2), device=heatmap.device)
        vals, idxs = torch.topk(flat, k)
        ys = idxs // heatmap.shape[1]
        xs = idxs % heatmap.shape[1]
        points = torch.stack([xs, ys], dim=1).float().unsqueeze(1)
        points = points * (self.model.image_size // heatmap.shape[0])
        return points

    def _save_heatmap_debug(
        self,
        inference_state,
        frame_idx,
        input_points,
        heatmap_predictions,
        radius=4,
    ):
        res_path = inference_state["res_path"]
        debug_dir = Path(res_path) / "heatmap_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_path = inference_state["frame_paths"][frame_idx]
        image = read_image(str(frame_path), return_np=True)
        overlay = image.copy()

        points = input_points.detach().cpu().numpy().astype(np.float32)
        pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]
        resized_h, resized_w = inference_state["resized_image_size"]
        scale_x = inference_state["video_width"] / resized_w
        scale_y = inference_state["video_height"] / resized_h
        crop_box = inference_state.get("crop_box", (0, 0, 0, 0))
        x_offset, y_offset = crop_box[0], crop_box[1]

        for point in points:
            x, y = point[0]
            x = (x - pad_left) * scale_x + x_offset
            y = (y - pad_top) * scale_y + y_offset
            x = int(np.clip(x, 0, image.shape[1] - 1))
            y = int(np.clip(y, 0, image.shape[0] - 1))
            cv2.circle(overlay, (x, y), radius, (255, 0, 0), -1)

        blended = cv2.addWeighted(image, 1.0, overlay, 0.6, 0)
        cv2.imwrite(str(debug_dir / f"heatmap_{frame_idx:03d}.png"), blended)

        heatmap = heatmap_predictions.detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap = (heatmap * 255).astype(np.uint8)
        pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]
        pad_right = inference_state["model_image_size"] - pad_right
        pad_bottom = inference_state["model_image_size"] - pad_bottom
        heatmap = heatmap[pad_top:pad_bottom, pad_left:pad_right]
        heatmap = cv2.resize(
            heatmap,
            (inference_state["video_width"], inference_state["video_height"]),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_full = np.zeros(image.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = crop_box
        heatmap_full[y0:y1, x0:x1] = heatmap
        heatmap_color = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(
            str(debug_dir / f"heatmap_raw_{frame_idx:03d}.png"), heatmap_overlay
        )
    
    def _postprocess_divisions(self, tiled_states, tracking_results, global_state):
        """Post-process to detect divisions across crops with ±1 frame tolerance.
        
        Uses local_to_global mappings (computed during main loop) to translate
        each crop's parent_ids into global IDs. For each division where the parent
        maps to a global cell and a daughter maps to a "new" cell, confirm it.
        """
        res_track = global_state["res_track"]
        res_path = global_state["res_path"]
        if len(res_track) == 0:
            return
        
        next_id = int(res_track[:, 0].max()) + 1
        
        # Collect new cells: parent_id == 0 and starts after frame 0
        new_cells = {}  # {cell_id: (row_idx, start_frame)}
        for row_idx in range(len(res_track)):
            cell_id, start_frame, _, parent_id = res_track[row_idx].astype(int)
            if parent_id == 0 and start_frame > 0:
                new_cells[int(cell_id)] = (row_idx, int(start_frame))
        
        if not new_cells:
            return
        
        # Find all divisions across all crops using local_to_global mapping
        # {(div_frame, global_parent): set of global daughter IDs}
        # Also store local daughter IDs for cross-frame matching
        # {(div_frame, global_parent): (set of global daughters, crop_idx, set of local daughters)}
        divisions = {}
        division_local_info = {}  # {(div_frame, global_parent): (crop_idx, set of local_daughters)}
        for frame_idx in range(len(tracking_results)):
            for crop_idx, state in enumerate(tiled_states):
                if frame_idx not in state.get("obj_ids", {}) or frame_idx not in state.get("parent_ids", {}):
                    continue
                l2g = state.get("local_to_global", {}).get(frame_idx, {})
                local_ids = state["obj_ids"][frame_idx].cpu().numpy()
                local_parents = state["parent_ids"][frame_idx].cpu().numpy()
                
                # Group global daughters by global parent
                # If local parent not in current frame mapping, check previous frame
                prev_l2g = state.get("local_to_global", {}).get(frame_idx - 1, {}) if frame_idx > 0 else {}
                parent_to_daughters = {}
                parent_to_local_daughters = {}  # Store local IDs too
                # First pass: collect daughters by local parent
                local_parent_to_daughters = {}
                for lid, lpid in zip(local_ids, local_parents, strict=False):
                    if int(lpid) == 0:
                        continue
                    local_parent_to_daughters.setdefault(int(lpid), []).append((int(lid), l2g.get(int(lid))))
                
                # Second pass: resolve global parents
                for lpid, daughters in local_parent_to_daughters.items():
                    if len(daughters) != 2:
                        continue  # Need at least 2 daughters for division
                    global_parent = l2g.get(lpid)
                    # If not found in current frame, try previous frame
                    if global_parent is None and frame_idx > 0:
                        global_parent = prev_l2g.get(lpid)
                    
                    if global_parent is not None:
                        for lid, gid in daughters:
                            if gid is not None:
                                parent_to_daughters.setdefault(global_parent, set()).add(gid)
                                parent_to_local_daughters.setdefault(global_parent, set()).add(lid)
                
                for global_parent, daughters in parent_to_daughters.items():
                    local_daughters_set = parent_to_local_daughters.get(global_parent, set())
                    # Check if there are 2+ local daughters (even if they map to same global cell)
                    if len(local_daughters_set) == 2:
                        key = (frame_idx, global_parent)
                        if key not in divisions:
                            divisions[key] = set()
                        divisions[key].update(daughters)
                        # Store local info for cross-frame matching
                        division_local_info[key] = (crop_idx, local_daughters_set)
        
        # For each division, also check the next frame's mapping to find new cells
        # This handles the case where the global mask hasn't split yet at the division frame
        # Check ALL crops at next frame, not just the detecting crop (daughter might be in different crop)
        for (div_frame, global_parent), daughter_ids in list(divisions.items()):
            next_frame = div_frame + 1
            if next_frame < len(tracking_results):
                div_crop_idx, local_daughters = division_local_info.get((div_frame, global_parent), (None, set()))
                if div_crop_idx is not None:
                    # Check ALL crops at next frame (daughter might appear in different crop)
                    for check_crop_idx, state in enumerate(tiled_states):
                        next_l2g = state.get("local_to_global", {}).get(next_frame, {})
                        for local_daughter in local_daughters:
                            if local_daughter in next_l2g:
                                mapped_global = next_l2g[local_daughter]
                                if mapped_global in new_cells:
                                    divisions[(div_frame, global_parent)].add(mapped_global)
        
        if not divisions:
            return
        
        found_divisions = False
        processed_parents = set()
        
        for (div_frame, global_parent), daughter_ids in divisions.items():
            if global_parent in processed_parents:
                continue
            
            # Check if any daughter is a "new cell"
            matched_new_cell = None
            matched_row_idx = None
            for daughter_id in daughter_ids:
                if daughter_id in new_cells:
                    start_frame = new_cells[daughter_id][1]
                    frame_diff = abs(start_frame - div_frame)
                    if frame_diff <= 1 and daughter_id != global_parent:
                        matched_new_cell = daughter_id
                        matched_row_idx = np.where(res_track[:, 0] == daughter_id)[0][0]
                        break
            
            if matched_new_cell is None:
                continue
            
            # Verify parent existed at div_frame - 1
            prev_frame = div_frame - 1
            if prev_frame < 0:
                continue
            parent_prev_count = (tracking_results[prev_frame] == global_parent).sum()
            if parent_prev_count == 0:
                continue
            
            # Verify parent still exists at div_frame (continued as one daughter)
            parent_div_count = (tracking_results[div_frame] == global_parent).sum()
            if parent_div_count == 0:
                continue
            
            division_frame = new_cells[matched_new_cell][1]

            # Skip if global_parent is not in res_track
            parent_rows = np.where(res_track[:, 0] == global_parent)[0]
            if len(parent_rows) == 0:
                continue
            parent_row = parent_rows[0]

            # Skip if parent didn't exist before division (start_frame > division_frame - 1)
            if res_track[parent_row, 1] > division_frame - 1:
                continue

            # 1. Set parent for the new daughter
            res_track[matched_row_idx, 3] = global_parent
            del new_cells[matched_new_cell]
            
            # 2. Create new daughter ID for the continued parent
            new_daughter_id = next_id
            next_id += 1

            # 3. Truncate parent and add continued-daughter row
            old_end = int(res_track[parent_row, 2])
            res_track[parent_row, 2] = division_frame - 1
            new_row = np.array([[new_daughter_id, division_frame, old_end, global_parent]])
            res_track = np.concatenate([res_track, new_row], axis=0)
            
            # 4. Update downstream parent refs: parent -> new_daughter_id
            # Exclude matched_new_cell and new_daughter_id - they should keep global_parent as parent
            for r in range(len(res_track)):
                cell_id = int(res_track[r, 0])
                if cell_id == matched_new_cell or cell_id == new_daughter_id:
                    continue  # Keep global_parent as parent for these cells
                if int(res_track[r, 3]) == global_parent and int(res_track[r, 1]) >= division_frame:
                    res_track[r, 3] = new_daughter_id
            
            # 5. Relabel masks from division_frame onward
            for f in range(division_frame, len(tracking_results)):
                mask = tracking_results[f]
                mask[mask == global_parent] = new_daughter_id
                cv2.imwrite(str(res_path / f"mask{f:03d}.tif"), mask.astype(np.uint16))

            # Update l2g to reflect global_parent -> new_daughter_id
            # Multiple crops can have local IDs mapping to the same global cell
            for state in tiled_states:
                for f in range(division_frame, len(tracking_results)):
                    l2g_f = state["local_to_global"][f]
                    for lid, g in list(l2g_f.items()):
                        if g == global_parent:
                            l2g_f[lid] = new_daughter_id

            # Edge case: P->(A,B) then new_daughter_id->(C,D) next frame. Use local_to_global: get
            # local IDs for A and B, check their global mapping in next frame. If it matches {C,D}, collapse.
            div_crop, local_daughters = division_local_info.get((div_frame, global_parent), (None, set()))
            child_rows = np.where(res_track[:, 3] == new_daughter_id)[0]
            next_div_frame = division_frame + 1
            if div_crop is not None and len(child_rows) == 2 and int(res_track[child_rows[0], 1]) == next_div_frame:
                gc1, gc2 = int(res_track[child_rows[0], 0]), int(res_track[child_rows[1], 0])
                if len(local_daughters) == 2:
                    l2g_div = tiled_states[div_crop].get("local_to_global", {}).get(division_frame, {})
                    l2g_next = tiled_states[div_crop].get("local_to_global", {}).get(next_div_frame, {})
                    local_a = local_b = None
                    for lid in local_daughters:
                        if l2g_div.get(lid) == matched_new_cell:
                            local_a = lid
                        elif l2g_div.get(lid) == new_daughter_id:
                            local_b = lid
                    if local_a is not None and local_b is not None:
                        global_a_next = l2g_next.get(local_a)
                        global_b_next = l2g_next.get(local_b)
                        if {global_a_next, global_b_next} == {gc1, gc2}:
                            map_a = global_a_next
                            res_track[matched_row_idx, 2] = res_track[child_rows[0] if map_a == gc1 else child_rows[1], 2]
                            new_d_row = np.where(res_track[:, 0] == new_daughter_id)[0][0]
                            res_track[new_d_row, 2] = res_track[child_rows[1] if map_a == gc1 else child_rows[0], 2]
                            res_track[res_track[:, 3] == gc1, 3] = matched_new_cell if map_a == gc1 else new_daughter_id
                            res_track[res_track[:, 3] == gc2, 3] = matched_new_cell if map_a == gc2 else new_daughter_id
                            res_track = np.delete(res_track, child_rows, axis=0)
                            for f in range(division_frame, len(tracking_results)):
                                mask = tracking_results[f]
                                if f > division_frame:
                                    mask[mask == gc1] = matched_new_cell if map_a == gc1 else new_daughter_id
                                    mask[mask == gc2] = matched_new_cell if map_a == gc2 else new_daughter_id
                                cv2.imwrite(str(res_path / f"mask{f:03d}.tif"), mask.astype(np.uint16))
                            # Update l2g so downstream code sees correct global IDs after collapse
                            g1_new = matched_new_cell if map_a == gc1 else new_daughter_id
                            g2_new = matched_new_cell if map_a == gc2 else new_daughter_id
                            # Update the local to global mapping
                            for state in tiled_states:
                                for f in range(next_div_frame, len(tracking_results)):
                                    l2g_f = state["local_to_global"][f]
                                    for lid, g in list(l2g_f.items()):
                                        if g == gc1:
                                            l2g_f[lid] = g1_new
                                        elif g == gc2:
                                            l2g_f[lid] = g2_new

            found_divisions = True
            processed_parents.add(global_parent)

        if found_divisions:
            global_state["res_track"] = res_track
            np.savetxt(res_path / "res_track.txt", res_track, fmt="%d")
    
    def save_ctc(self, track_mask, frame_idx, inference_state):
        res_path = inference_state["res_path"]

        cell_ids_track_mask = np.unique(track_mask)
        cell_ids_track_mask = cell_ids_track_mask[cell_ids_track_mask != 0]

        cell_ids = inference_state["obj_ids"][frame_idx].cpu().numpy()

        assert sorted(cell_ids_track_mask) == sorted(cell_ids), (
            "cell_ids_track_mask and cell_ids must be the same"
        )

        if len(cell_ids) > 0:
            assert max(cell_ids) < 65536, "cell_id must be less than 65536"

        cv2.imwrite(
            str(res_path / f"mask{frame_idx:03d}.tif"), track_mask.astype(np.uint16)
        )

        if not self.segment or self.aux_matching == "segment_then_aux_track":
            parent_ids_tensor = inference_state["parent_ids"].get(frame_idx)
            if parent_ids_tensor is not None and len(parent_ids_tensor) == len(cell_ids):
                parent_ids = parent_ids_tensor.cpu().numpy()
            else:
                parent_ids = np.zeros(len(cell_ids), dtype=np.int32)
            res_track = inference_state["res_track"]

            for cell_id, parent_id in zip(cell_ids, parent_ids, strict=False):
                if cell_id not in res_track[:, 0]:
                    res_track = np.concatenate(
                        [
                            res_track,
                            np.array([[cell_id, frame_idx, frame_idx, parent_id]]),
                        ],
                        axis=0,
                    )
                else:
                    assert res_track[res_track[:, 0] == cell_id, 2] == frame_idx - 1, (
                        "cell_id must be continuous"
                    )
                    res_track[res_track[:, 0] == cell_id, 2] = frame_idx

            np.savetxt(res_path / "res_track.txt", res_track, fmt="%d")

            inference_state["res_track"] = res_track

    def save_tracking_results(self, inference_state, tracking_results, alpha=0.3, crop_idx=None):
        res_path = inference_state["res_path"]

        if self.segment and self.aux_matching != "off":
            num_colors = 1000
        else:
            num_colors = (
                inference_state["max_obj_id"] + 1
            )  # Add 1 to account for 0-based indexing
        colors = np.random.randint(0, 255, (num_colors, 3))
        color_stack = np.zeros(
            (
                len(tracking_results),
                inference_state["video_height"],
                inference_state["video_width"],
                3,
            ),
            dtype=np.uint8,
        )

        if "frame_paths" not in inference_state:
            raise ValueError("frame_paths is required to render tracking results.")

        for frame_idx, track_mask in enumerate(tracking_results):
            frame_path = inference_state["frame_paths"][frame_idx]
            img = read_image(str(frame_path), return_np=True)
            
            # Crop image if crop_box is specified (for crop movies)
            if "crop_box" in inference_state:
                x0, y0, x1, y1 = inference_state["crop_box"]
                img = img[y0:y1, x0:x1]

            # Create a colored overlay image
            overlay = np.zeros_like(img)

            cell_ids = np.unique(track_mask)
            cell_ids = cell_ids[cell_ids != 0]  # Exclude background (0)

            # Add colored masks for each cell
            for cell_id in cell_ids:
                mask = track_mask == cell_id
                overlay[mask] = colors[cell_id]

            # Blend original image with colored overlay
            color_stack[frame_idx] = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            for cell_id in cell_ids:
                mask = track_mask == cell_id
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue

                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))

                label = str(cell_id)
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                origin_x = int(centroid_x - text_w / 2)
                origin_y = int(centroid_y + text_h / 2)
                cv2.putText(
                    color_stack[frame_idx],
                    label,
                    (origin_x, origin_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (255, 255, 255),  # black color
                    1,  # Line thickness
                    cv2.LINE_AA,
                )

            if not self.segment or self.aux_matching == "segment_then_aux_track":
                # Use res_track to find divisions (more reliable than inference_state)
                res_track = inference_state.get("res_track")
                if res_track is not None and len(res_track) > 0:
                    # Find all cells that exist in this frame
                    cells_in_frame = []
                    cell_info = {}  # {cell_id: (start_frame, end_frame, parent_id)}
                    for row in res_track:
                        cell_id, start_frame, end_frame, parent_id = row.astype(int)
                        if start_frame <= frame_idx <= end_frame:
                            cells_in_frame.append((cell_id, parent_id))
                            cell_info[cell_id] = (start_frame, end_frame, parent_id)
                    
                    # Group by parent_id
                    parent_to_daughters = {}
                    for cell_id, parent_id in cells_in_frame:
                        if parent_id != 0:
                            if parent_id not in parent_to_daughters:
                                parent_to_daughters[parent_id] = []
                            parent_to_daughters[parent_id].append(cell_id)
                    
                    # Draw line between daughter cells only on the division frame
                    # (when both daughters start at this frame)
                    for parent_id, dau_cell_ids in parent_to_daughters.items():
                        if len(dau_cell_ids) == 2:
                            # Check if this is the division frame (both daughters start here)
                            dau1_start = cell_info[dau_cell_ids[0]][0]
                            dau2_start = cell_info[dau_cell_ids[1]][0]
                            if dau1_start == frame_idx and dau2_start == frame_idx:
                                # Get centroids of both daughter cells
                                mask1 = track_mask == dau_cell_ids[0]
                                y1, x1 = np.where(mask1)
                                if len(y1) > 0:
                                    centroid1_y = int(np.mean(y1))
                                    centroid1_x = int(np.mean(x1))

                                    mask2 = track_mask == dau_cell_ids[1]
                                    y2, x2 = np.where(mask2)
                                    if len(y2) > 0:
                                        centroid2_y = int(np.mean(y2))
                                        centroid2_x = int(np.mean(x2))

                                        # Draw line connecting centroids
                                        cv2.line(
                                            color_stack[frame_idx],
                                            (centroid1_x, centroid1_y),
                                            (centroid2_x, centroid2_y),
                                            (0, 0, 0),  # Black color
                                            1,
                                        )  # Line thickness

            # Add frame number to top of frame
            cv2.putText(
                color_stack[frame_idx],
                f"{frame_idx:03}",
                (0, 15),  # Position in top-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 255, 255),  # White color
                1,  # Line thickness
                cv2.LINE_AA,
            )

        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        mode = "segment" if self.segment else "track"
        video_filename = f"pred_{mode}_video.mp4"
        if crop_idx is not None:
            video_filename = f"pred_{mode}_video_crop_{crop_idx}.mp4"
        out = cv2.VideoWriter(
            str(res_path / video_filename),
            fourcc,
            10.0,  # 10 fps
            (inference_state["video_width"], inference_state["video_height"]),
        )

        for frame in color_stack:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
