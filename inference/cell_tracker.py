from pathlib import Path
from typing import Optional

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
        obj_score_thresh: float = 0,
        pred_iou_thresh: float = 0.7,
        div_obj_score_thresh: float = 0,
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
        crop_reassign_iou_thresh: float = 0.3,
        save_crop_movies: bool = False,
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
        """
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
        self.obj_score_thresh = obj_score_thresh
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
        if height <= self.resize_threshold:
            ys = [0]
        else:
            ys = list(range(0, max(1, height - target_size + 1), stride))
        if width <= self.resize_threshold:
            xs = [0]
        else:
            xs = list(range(0, max(1, width - target_size + 1), stride))
        if ys[-1] != height - target_size:
            ys.append(max(0, height - target_size))
        if xs[-1] != width - target_size:
            xs.append(max(0, width - target_size))
        boxes = []
        for y0 in ys:
            for x0 in xs:
                x1 = min(width, x0 + target_size)
                y1 = min(height, y0 + target_size)
                if x1 - x0 < target_size:
                    x0 = max(0, x1 - target_size)
                if y1 - y0 < target_size:
                    y0 = max(0, y1 - target_size)
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

    def _assign_cells_to_crops(self, tiled_states, crop_centers, crop_masks, full_mask, frame_idx=0):
        """Assign cells to crops for tracking continuity.
        
        Frame 0: Assign all cells to nearest crop (initial assignment)
        Frame 1+: Reassign only if cell overlaps (IoU > threshold) with cell in another crop
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_centers: List of (center_x, center_y) tuples for each crop
            crop_masks: List of crop masks (numpy arrays)
            full_mask: Already merged full mask
            frame_idx: Current frame index
            
        Returns:
            crop_assignments: List of sets, one per crop, containing assigned cell IDs
        """
        crop_assignments = [set() for _ in tiled_states]
        if not tiled_states:
            return crop_assignments

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

        # Frame 0: Assign all cells to nearest crop
        if frame_idx == 0:
            assignments = self._assign_to_nearest_crop(centroids, crop_centers)
            for obj_id, assigned in zip(obj_id_list, assignments, strict=False):
                crop_assignments[assigned].add(obj_id)
            return crop_assignments
        
        # Frame 1+: only reassign if cell overlaps (IoU > threshold) with cell in another crop
        # First, determine which crop each cell was previously assigned to
        cell_to_prev_crop = {}
        for crop_idx, state in enumerate(tiled_states):
            # Check if this crop has memory for the cell (indicates it was tracking it)
            if "memory_dict" in state:
                for obj_id in state["memory_dict"].keys():
                    if obj_id == "mask_mem_pos_enc":
                        continue
                    obj_id_int = int(obj_id)
                    if obj_id_int not in cell_to_prev_crop:
                        cell_to_prev_crop[obj_id_int] = crop_idx
            
            # Also check previous frame's obj_ids
            if "obj_ids" in state and state["obj_ids"] is not None:
                if frame_idx - 1 in state["obj_ids"]:
                    prev_obj_ids = state["obj_ids"][frame_idx - 1].cpu().numpy()
                    for obj_id in prev_obj_ids:
                        obj_id_int = int(obj_id)
                        if obj_id_int not in cell_to_prev_crop:
                            cell_to_prev_crop[obj_id_int] = crop_idx
        
        # For each cell that appears in the current mask
        for obj_id in obj_ids:
            obj_id_int = int(obj_id)
            prev_crop = cell_to_prev_crop.get(obj_id_int)
            
            if prev_crop is None:
                # New cell (not tracked before) - assign to nearest crop
                centroid = obj_id_to_centroid[obj_id_int]
                assignments = self._assign_to_nearest_crop([centroid], crop_centers)
                crop_assignments[assignments[0]].add(obj_id_int)
            else:
                # Existing cell - only reassign if other crop has detected the SAME cell ID
                # This prevents reassigning daughter cells to crops that haven't predicted the division yet
                target_crop = prev_crop  # Default: keep in original crop
                
                # Check for reassignment (cell appears in mask, so obj_id_to_centroid exists)
                if obj_id_int in obj_id_to_centroid:
                    current_cell_mask = (full_mask == obj_id_int)
                    
                    # Check if other crops have detected the same cell ID
                    for crop_idx in range(len(tiled_states)):
                        if crop_idx == prev_crop:
                            continue
                        
                        # Get all cells in this crop
                        crop_mask = crop_masks[crop_idx]
                        crop_cell_ids = np.unique(crop_mask)
                        crop_cell_ids = crop_cell_ids[crop_cell_ids != 0]
                        
                        # Only reassign if the other crop has the SAME cell ID
                        # (i.e., it has already detected/predicted this cell)
                        if obj_id_int in crop_cell_ids:
                            # Other crop has detected this cell - check IoU to confirm it's the same
                            other_cell_mask = (crop_mask == obj_id_int)
                            # Get the crop box to extract the relevant region
                            crop_box = tiled_states[crop_idx]["crop_box"]
                            x0, y0, x1, y1 = crop_box
                            
                            # Extract region from full_mask for current cell
                            current_cell_in_crop = current_cell_mask[y0:y1, x0:x1]
                            
                            # Compute IoU
                            iou = self._compute_mask_iou(current_cell_in_crop, other_cell_mask)
                            
                            if iou > self.crop_reassign_iou_thresh:
                                target_crop = crop_idx
                                break
                
                # Assign to target crop (either original or new if reassigned)
                # If other crop hasn't detected this cell yet, keep it in original crop
                # This ensures continuity - cell stays in its original crop even if not in mask
                crop_assignments[target_crop].add(obj_id_int)

        return crop_assignments

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
        
        return full_mask

    def _overlay_tracked_masks(self, tiled_states, crop_masks):
        """Overlay full crop predictions for tracked cells.
        
        Unlike _merge_crop_masks which uses non-overlapping regions, this overlays
        the full prediction from each crop. Cells are already filtered to assigned ones.
        
        Args:
            tiled_states: List of inference states, one per crop
            crop_masks: List of filtered crop masks (only assigned cells)
            
        Returns:
            Merged full_mask with full cell predictions overlaid
        """
        full_height = tiled_states[0]["full_video_height"]
        full_width = tiled_states[0]["full_video_width"]
        full_mask = np.zeros((full_height, full_width), dtype=np.uint16)
        
        # Overlay each crop's full prediction
        for state, crop_mask in zip(tiled_states, crop_masks, strict=False):
            x0, y0, x1, y1 = state["crop_box"]
            # Overlay full crop mask (only non-zero pixels to avoid overwriting)
            crop_region = full_mask[y0:y1, x0:x1]
            crop_mask_nonzero = crop_mask > 0
            crop_region[crop_mask_nonzero] = crop_mask[crop_mask_nonzero]
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
                
                # Check if crops share an edge (right-left or bottom-top)
                if full_x1_i == full_x0_j:  # i's right edge touches j's left edge
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
                for cell_id_i in cell_ids_i:
                    cell_in_overlap_i = (overlap_mask_i == cell_id_i)
                    
                    for cell_id_j in cell_ids_j:
                        cell_in_overlap_j = (overlap_mask_j == cell_id_j)
                        
                        # Compute IoU in the overlapping area using original crop masks
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

        # Initialize crop_assignments storage in each state
        for state in tiled_states:
            state["crop_assignments"] = {}
        
        # Store crop tracking results if saving crop movies
        crop_tracking_results = [[] for _ in tiled_states] if self.save_crop_movies else None
        
        for frame_idx in tqdm(range(num_frames), desc="propagate in video"):
            crop_masks = []
            for gen in generators:
                _, state, track_mask = next(gen)
                crop_masks.append(track_mask)
            
            # Store crop masks for movie saving
            if self.save_crop_movies:
                for crop_idx, crop_mask in enumerate(crop_masks):
                    crop_tracking_results[crop_idx].append(crop_mask)

            # Frame 0: Segmentation - merge all crop masks, then assign cells to crops
            if frame_idx == 0:
                full_mask = self._merge_crop_masks(tiled_states, crop_masks)
            else:
                # Frame 1+: Tracking - use previous frame's assignments to filter current predictions
                prev_assignments = [state["crop_assignments"].get(frame_idx - 1, set()) for state in tiled_states]
                
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
                        # Check if assigned cell exists in current crop
                        cell_mask = (track_mask == assigned_cell_id)
                        if cell_mask.sum() > 0:
                            filtered_mask[cell_mask] = assigned_cell_id
                        else:
                            # Cell doesn't exist - check for daughters (cells with this as parent)
                            daughter_ids = [obj_id for obj_id, parent_id in parent_map.items() 
                                          if parent_id == assigned_cell_id]
                            for daughter_id in daughter_ids:
                                daughter_mask = (track_mask == daughter_id)
                                if daughter_mask.sum() > 0:
                                    filtered_mask[daughter_mask] = daughter_id
                    
                    filtered_masks.append(filtered_mask)
                
                # Merge tracked masks - overlay full predictions from each crop
                tracked_mask = self._overlay_tracked_masks(tiled_states, filtered_masks)
                
                # Merge all crops (like frame 0) to get segmentation mask for new cells
                seg_mask = self._merge_crop_masks(tiled_states, crop_masks)
                
                # Overlay: add new cells from seg_mask that don't overlap with tracked cells
                full_mask = tracked_mask.copy()
                seg_cell_ids = np.unique(seg_mask)
                seg_cell_ids = seg_cell_ids[seg_cell_ids != 0]
                
                for seg_cell_id in seg_cell_ids:
                    seg_cell_mask = (seg_mask == seg_cell_id)
                    # Check overlap with tracked cells
                    overlap_mask = np.logical_and(seg_cell_mask, tracked_mask > 0)
                    overlap_ratio = overlap_mask.sum() / seg_cell_mask.sum() if seg_cell_mask.sum() > 0 else 0
                    
                    if overlap_ratio >= 0.3:  # Significant overlap - merge into tracked cell
                        # Find which tracked cell it overlaps with most
                        overlapping_cells = tracked_mask[overlap_mask]
                        if len(overlapping_cells) > 0:
                            tracked_cell_id = np.bincount(overlapping_cells).argmax()
                            # Merge seg_cell into tracked cell
                            full_mask[seg_cell_mask] = tracked_cell_id
                    else:  # Low overlap - add as new cell
                        # Only add pixels that aren't already in tracked_mask
                        new_cell_mask = np.logical_and(seg_cell_mask, tracked_mask == 0)
                        full_mask[new_cell_mask] = seg_cell_id
                
            if frame_idx < num_frames:
                # Compute new assignments for current frame (to use in next frame)
                crop_assignments = self._assign_cells_to_crops(
                    tiled_states, crop_centers, crop_masks, full_mask, frame_idx=frame_idx
                )
                # Store assignments for next frame
                for crop_idx, state in enumerate(tiled_states):
                    state["crop_assignments"][frame_idx] = crop_assignments[crop_idx]
            
            tracking_results.append(full_mask)

            obj_ids = np.unique(full_mask)
            obj_ids = obj_ids[obj_ids != 0]
            global_state["obj_ids"][frame_idx] = torch.tensor(
                obj_ids, device=self.device, dtype=torch.int32
            )
            parent_ids = np.zeros_like(obj_ids, dtype=np.int32)
            parent_map = {}
            for state in tiled_states:
                if state["obj_ids"] is None or frame_idx not in state["obj_ids"]:
                    continue
                local_ids = state["obj_ids"][frame_idx].cpu().numpy()
                local_parents = state["parent_ids"][frame_idx].cpu().numpy()
                for obj_id, parent_id in zip(local_ids, local_parents, strict=False):
                    parent_map[int(obj_id)] = int(parent_id)
            for i, obj_id in enumerate(obj_ids):
                parent_ids[i] = parent_map.get(int(obj_id), 0)
            global_state["parent_ids"][frame_idx] = torch.tensor(
                parent_ids, device=self.device, dtype=torch.int32
            )

            self.save_ctc(full_mask, frame_idx, global_state)

        max_obj_id = 0
        for state in tiled_states:
            global_state_value = state.get("global_id_state", {}).get("value", 0)
            max_obj_id = max(max_obj_id, int(global_state_value))
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
                    "res_path": res_path / f"crop_{crop_idx}",
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
                
                crop_state["res_path"].mkdir(parents=True, exist_ok=True)
                self.save_tracking_results(crop_state, crop_results, crop_idx=crop_idx)

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
                current_out, sam_outputs, high_res_features, pix_feat = (
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
                            current_out, sam_outputs, high_res_features, pix_feat = (
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
            (object_score_logits_dict["post_div"][:, 0] > self.obj_score_thresh)
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
            object_score_logits_dict["post_div"][:, 0] > self.obj_score_thresh
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

        if not self.segment:
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

        if not self.segment:
            parent_ids = inference_state["parent_ids"][frame_idx].cpu().numpy()
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

        if self.segment:
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

            if not self.segment:
                parent_ids = inference_state["parent_ids"][frame_idx].cpu().numpy()
                parent_ids_unique = np.unique(parent_ids)
                parent_ids_unique = parent_ids_unique[parent_ids_unique != 0]

                for parent_id in parent_ids_unique:
                    dau_cell_ids = (
                        inference_state["obj_ids"][frame_idx][parent_ids == parent_id]
                        .cpu()
                        .numpy()
                    )

                    # Draw line between daughter cells
                    if len(dau_cell_ids) == 2:
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
