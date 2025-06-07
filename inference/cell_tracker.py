import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import batched_nms
import torch.nn.functional as F
import cv2

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.amg import (
    batch_iterator, 
    batched_mask_to_box, 
    calculate_stability_score, 
    MaskData, 
    area_from_rle, 
    box_xyxy_to_xywh, 
    mask_to_rle_pytorch, 
    coco_encode_rle, 
    rle_to_mask
)
from sam2.utils.misc import load_video_frames
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
        stability_score_thresh: float = 0,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.5,
        max_hole_area: int = 0,
        max_sprinkle_area: int = 0,
        mask_threshold: float = 0.0,
        segment: bool = False,
        use_heatmap: bool = False,
        min_mask_area: int = 30,
    ) -> None:
        """
        Using a SAM 2 model, generates and tracks masks for an entire video.
        Generates a grid of point prompts over the first frame, then tracks the detected cells
        throughout the video.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          segment (bool): Whether to segment or track.
        """

            
        self.model = model
        self.model.sam_mask_decoder.pred_iou_thresh = pred_iou_thresh
        self.model.sam_mask_decoder.obj_score_thresh = obj_score_thresh
        self.model.sam_mask_decoder.div_obj_score_thresh = div_obj_score_thresh
        self.device = model.device

        if self.device.type == 'cpu':
            min_mask_region_area = 0

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.obj_score_thresh = obj_score_thresh

        self.pred_iou_thresh = pred_iou_thresh
        self.obj_score_thresh = obj_score_thresh
        self.div_obj_score_thresh = div_obj_score_thresh
        self.segment = segment
        self.use_heatmap = use_heatmap
        self.min_mask_area = min_mask_area  

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        res_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        max_frame_num_to_track=None,
    ):
        """Initialize an inference state."""
        compute_device = self.model.device  # device of the model
        images, video_height, video_width, resized_image_size, padding = load_video_frames(
            video_path=video_path,
            image_size=self.model.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
            transforms=self._transforms,
        )
        inference_state = {}
        inference_state["res_path"] = res_path
        inference_state["video_path"] = video_path
        inference_state["res_track"] = np.zeros((0,4))
        inference_state["resized_image_size"] = resized_image_size
        inference_state["model_image_size"] = self.model.image_size
        inference_state["padding"] = padding
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["parent_ids"] = {}
        inference_state["max_frame_num_to_track"] = max_frame_num_to_track
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_ids"] = None
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        inference_state["memory_dict"] = {"mask_mem_pos_enc": None}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    def predict(self, video_path, res_path, offload_video_to_cpu=True, offload_state_to_cpu=False, max_frame_num_to_track=None):
        """
        Predict and track cells throughout a video.
        
        Args:
            video_path: Path to the video file
            offload_video_to_cpu: Whether to offload video frames to CPU to save GPU memory
            offload_state_to_cpu: Whether to offload inference state to CPU
            
        Returns:
            Dictionary of tracking results with frame indices as keys
        """
        # Initialize the video state
        inference_state = self.init_state(
            video_path=video_path,
            res_path=res_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            max_frame_num_to_track=max_frame_num_to_track
        )
        
        if not self.use_heatmap:
            # Generate points for the first frame or whole video if segment is True
            inference_state = self.generate_proportional_point_grid(inference_state)
        
        # Detect and track the detected cells through the video
        tracking_results = self.track_cells(inference_state)
        
        self.save_tracking_results(inference_state, tracking_results)
        
        return tracking_results

    def generate_proportional_point_grid(self, inference_state):
        """
        Generate a grid of (x, y) points with density proportional to the image size.
        
        Args:
            inference_state: The video inference state
        
        Returns:
            points: torch.Tensor of shape [N, 2] where each row is (x, y)
            labels: torch.Tensor of shape [N] with all 1s (foreground)
        """
        H, W = inference_state["video_height"], inference_state["video_width"]
        resized_H, resized_W = inference_state["resized_image_size"]
        scale_factor_y = (resized_H / self.model.image_size)  # preserve relative density
        scale_factor_x = (resized_W / self.model.image_size)  # preserve relative density

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
            
        points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)[:,None]

        # Convert points to tensor
        points = torch.tensor(points, device=self.device, dtype=torch.float32)  # [N, 2]
        
        # Create corresponding labels tensor (all foreground)
        labels = torch.ones((len(points), 1), dtype=torch.int, device=self.device)  # [N]
        
        # Save points and labels in inference_state for later reference
        inference_state["point_inputs"][0] = {"point_coords": points, "point_labels": labels}

        if self.segment:
            for i in range(1,len(inference_state["images"])):
                inference_state["point_inputs"][i] = {"point_coords": points, "point_labels": labels}

        return inference_state

    def track_cells(self, inference_state):
        """
        Track the detected cells throughout the video.
        
        Args:
            inference_state: The video inference state
            
        Returns:
            Dictionary of tracking results with frame indices as keys
        """
        tracking_results = []
        
        # Start propagation through the video
        for frame_idx, inference_state, track_mask in self.propagate_in_video(inference_state):

            tracking_results.append(track_mask)

            self.save_ctc(track_mask, frame_idx, inference_state)
        
        return tracking_results

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=0,
    ):
        """Propagate the input points across frames to track in the entire video."""
        num_frames = inference_state["num_frames"]
        max_frame_num_to_track = inference_state["max_frame_num_to_track"]
        
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames

        end_frame_idx = min(
            start_frame_idx + max_frame_num_to_track, num_frames - 1
        )
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):     

            if frame_idx == 0 or self.segment:
                if self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(inference_state, frame_idx)
                    inference_state["point_inputs"][frame_idx] = {"point_coords": input_points, "point_labels": point_labels}
                tracking_object_ids = None
                batch_size = inference_state["point_inputs"][frame_idx]['point_coords'].shape[0]
                is_init_cond_frame = True
            else:
                tracking_object_ids = inference_state["obj_ids"][frame_idx-1]
                batch_size = len(tracking_object_ids)       
                is_init_cond_frame = False

            if batch_size == 0:
                inference_state["obj_ids"][frame_idx] = torch.zeros(0, device=self.device, dtype=torch.int32)
                inference_state["parent_ids"][frame_idx] = torch.zeros(0, device=self.device, dtype=torch.int32)
                track_mask = np.zeros((inference_state["video_height"], inference_state["video_width"]), dtype=np.uint16)
                yield frame_idx, inference_state, track_mask

            else:
                # Retrieve image features (only need to compute once for all objects)
                (
                    _,
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._get_image_feature(inference_state, frame_idx, batch_size)
                
                # Run the core tracking step
                current_out, sam_outputs, high_res_features, pix_feat = self.model._track_step(
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=inference_state["point_inputs"].get(frame_idx, None),
                    mask_inputs=None,
                    num_frames=inference_state["num_frames"],
                    prev_sam_mask_logits=None,
                    tracking_object_ids=tracking_object_ids,
                    memory_dict=inference_state["memory_dict"],
                )


                # update cell tracks
                inference_state, track_mask = self.update_cell_tracks(inference_state, frame_idx, sam_outputs, current_out, tracking_object_ids)

                if not self.segment and frame_idx > 0 and self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(inference_state, frame_idx)
                    input_points_copy = input_points.clone()

                    pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]
                    input_points[:,0,0] -= pad_left
                    input_points[:,0,1] -= pad_top

                    input_points[:,0,0] = (input_points[:,0,0] * (inference_state["video_width"] / inference_state["resized_image_size"][1]))
                    input_points[:,0,1] = (input_points[:,0,1] * (inference_state["video_height"] / inference_state["resized_image_size"][0]))

                    # Convert to numpy and int
                    input_points_np = input_points.cpu().numpy().astype(np.int32)
                    track_cell_ids = track_mask[input_points_np[:,0,1], input_points_np[:,0,0]]

                    # Find indices where track_cell_ids is 0 (background)
                    background_point_indices = np.where(track_cell_ids == 0)[0]

                    if len(background_point_indices) > 0:
                        input_points = input_points_copy[background_point_indices]
                        point_labels = point_labels[background_point_indices]

                        inference_state["point_inputs"][frame_idx] = {"point_coords": input_points, "point_labels": point_labels}
                        batch_size = input_points.shape[0]

                        # Retrieve image features (only need to compute once for all objects)
                        (
                            _,
                            _,
                            current_vision_feats,
                            current_vision_pos_embeds,
                            feat_sizes,
                        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
                        
                        # Run the core tracking step
                        current_out, sam_outputs, high_res_features, pix_feat = self.model._track_step(
                            is_init_cond_frame=True,
                            current_vision_feats=current_vision_feats,
                            current_vision_pos_embeds=current_vision_pos_embeds,
                            feat_sizes=feat_sizes,
                            point_inputs=inference_state["point_inputs"].get(frame_idx, None),
                            mask_inputs=None,
                            num_frames=inference_state["num_frames"],
                            prev_sam_mask_logits=None,
                            tracking_object_ids=None,
                            memory_dict=inference_state["memory_dict"],
                        )

                        inference_state, detected_mask = self.update_cell_tracks(inference_state, frame_idx, sam_outputs, current_out, heatmap_input=True)

                        if detected_mask.sum() > 0:
                            detected_cells = np.unique(detected_mask)
                            detected_cells = detected_cells[detected_cells != 0]
                            
                            if len(inference_state["lost_obj_ids"][frame_idx]) > 0:
                                lost_obj_ids = inference_state["lost_obj_ids"][frame_idx]
                                lost_high_res_masks = inference_state["lost_high_res_masks"][frame_idx]

                                # Calculate IoU between each detected cell and lost cell
                                ious = np.zeros((len(detected_cells), len(lost_obj_ids)))
                                for i, detected_id in enumerate(detected_cells):
                                    detected_mask_binary = detected_mask == detected_id
                                    for j, lost_id in enumerate(lost_obj_ids):
                                        intersection = np.logical_and(detected_mask_binary, lost_high_res_masks[j]).sum()
                                        union = np.logical_or(detected_mask_binary, lost_high_res_masks[j]).sum()
                                        ious[i,j] = intersection / union if union > 0 else 0

                                # Find the lost cell with the highest IoU for each detected cell
                                max_ious = np.max(ious, axis=1)
                                lost_cell_indices = np.argmax(ious, axis=1)
                                
                                # Process each detected cell in order of IoU
                                sorted_indices = np.argsort(-max_ious)  # Sort by descending IoU
                                processed_lost_cells = set()
                                cells_to_remove = set()  # Track which cells to remove
                                
                                for idx in sorted_indices:
                                    if max_ious[idx] > 0:  # If cell has positive object score, it is assumed to be match if there is any overlap
                                        detected_cell_id = int(detected_cells[idx])
                                        lost_idx = lost_cell_indices[idx]
                                        lost_cell_id = int(lost_obj_ids[lost_idx])
                                        
                                        # Skip if this lost cell was already matched
                                        if lost_cell_id in processed_lost_cells:
                                            continue
                                            
                                        if frame_idx - 1 in inference_state["memory_dict"][lost_cell_id]['frame_idx']:
                                            detected_mask[detected_mask == detected_cell_id] = lost_cell_id
                                            inference_state["memory_dict"][lost_cell_id]['mask_mem_features'] = torch.cat((inference_state["memory_dict"][lost_cell_id]['mask_mem_features'], inference_state["memory_dict"][detected_cell_id]['mask_mem_features']), dim=0)
                                            inference_state["memory_dict"][lost_cell_id]['obj_ptr'] = torch.cat((inference_state["memory_dict"][lost_cell_id]['obj_ptr'], inference_state["memory_dict"][detected_cell_id]['obj_ptr']), dim=0)
                                            inference_state["memory_dict"][lost_cell_id]['frame_idx'].append(frame_idx)
                                            inference_state["obj_ids"][frame_idx][inference_state["obj_ids"][frame_idx] == detected_cell_id] = lost_cell_id
                                            
                                            cells_to_remove.add(detected_cell_id)  # Mark for removal
                                            processed_lost_cells.add(lost_cell_id)
                                            
                                            del inference_state["memory_dict"][detected_cell_id]

                                # Remove the cells after processing all matches
                                detected_cells = detected_cells[~np.isin(detected_cells, list(cells_to_remove))]

                            # Handle remaining detected cells
                            for detected_cell_id in detected_cells:
                                # Get binary mask for current detected cell
                                detected_mask_binary = detected_mask == detected_cell_id
                                
                                # Get all unique track IDs that overlap with this detected cell
                                overlapping_track_ids = np.unique(track_mask[detected_mask_binary])
                                overlapping_track_ids = overlapping_track_ids[overlapping_track_ids > 0]  # Remove background (0)
                                
                                if len(overlapping_track_ids) > 0:
                                    # Calculate IoU with each overlapping track
                                    best_iou = 0
                                    best_track_id = None
                                    
                                    for track_id in overlapping_track_ids:
                                        track_mask_binary = track_mask == track_id
                                        intersection = np.logical_and(detected_mask_binary, track_mask_binary).sum()
                                        union = np.logical_or(detected_mask_binary, track_mask_binary).sum()
                                        iou = intersection / union if union > 0 else 0
                                        
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_track_id = track_id
                                    
                                    if best_iou > 0.05:  # If there's any overlap, assume it's the same cell
                                        # Update the detected mask to use the best matching track ID
                                        detected_mask[detected_mask_binary] = best_track_id
                                        inference_state["memory_dict"][best_track_id]['mask_mem_features'][-1] = inference_state["memory_dict"][detected_cell_id]['mask_mem_features'][0]
                                        inference_state["memory_dict"][best_track_id]['obj_ptr'][-1] = inference_state["memory_dict"][detected_cell_id]['obj_ptr'][0]
                                        inference_state["parent_ids"][frame_idx] = inference_state["parent_ids"][frame_idx][inference_state["obj_ids"][frame_idx] != detected_cell_id]
                                        inference_state["obj_ids"][frame_idx] = inference_state["obj_ids"][frame_idx][inference_state["obj_ids"][frame_idx] != detected_cell_id]
                                        
                                        del inference_state["memory_dict"][detected_cell_id]
                                
                            track_mask[(detected_mask > 0) * (track_mask == 0)] = detected_mask[(detected_mask > 0) * (track_mask == 0)]

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

    def update_cell_tracks(self, inference_state, frame_idx, sam_outputs, current_out, tracking_object_ids=None, heatmap_input=False):
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
            mask = high_res_masks[i,0].cpu().numpy()
            mask_binary = mask > self.mask_threshold
            if mask_binary.any():
                # Find connected components
                num_labels, labels = cv2.connectedComponents(mask_binary.astype(np.uint8))
                if num_labels > 1:  # If there are multiple components
                    # Find sizes of all components
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    # Get label of largest component (excluding background label 0)
                    largest_label = unique_labels[1:][np.argmax(counts[1:])]
                    # Keep only largest component
                    mask_binary = (labels == largest_label)
                    save_masks[i,0][torch.from_numpy(mask_binary).to(high_res_masks.device)] = high_res_masks[i,0][torch.from_numpy(mask_binary).to(high_res_masks.device)]
                else:
                    save_masks[i,0][high_res_masks[i,0] > self.mask_threshold] = high_res_masks[i,0][high_res_masks[i,0] > self.mask_threshold]

        
                 # Get max scores across all masks for each pixel
        argmax_scores = torch.max(save_masks[:,0], dim=0)[1]  # shape: (H, W)
        # Count pixels for each mask index (excluding background)
        valid_mask = save_masks[:,0].sum(0) > 0
        valid_indices = argmax_scores[valid_mask]
        max_mask_area = torch.bincount(valid_indices.flatten(), minlength=len(save_masks))

        keep_tokens = (
            object_score_logits_dict["post_div"][:,0] > self.obj_score_thresh
            ) * (
                ious[:,0] > self.pred_iou_thresh
            ) * (
                max_mask_area > self.min_mask_area
            )
            
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=high_res_masks[keep_tokens].flatten(0, 1),
            save_masks=save_masks[keep_tokens].flatten(0, 1),
            iou_preds=ious[keep_tokens].flatten(0, 1),
            obj_scores=object_score_logits_dict["post_div"][keep_tokens].flatten(0,1),
            obj_ptr=obj_ptr[keep_tokens]
        )

        data["boxes"] = batched_mask_to_box(data["masks"] > self.mask_threshold)
        
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        ).sort()[0]

        data.filter(keep_by_nms)

        removed_indices = torch.nonzero(keep_tokens)[~torch.isin(torch.arange(keep_tokens.sum(), device=keep_tokens.device), keep_by_nms)]
        keep_tokens[removed_indices] = False

        # Store which cells are predicted to be objects but are not kept by NMS or iou score or mask threshold
        valid_next_frame_mask = object_score_logits_dict["post_div"][:,0] > self.obj_score_thresh

        if heatmap_input:
            obj_ids = torch.arange(inference_state["max_obj_id"]+1, inference_state["max_obj_id"]+1+data["masks"].shape[0], device=self.device, dtype=torch.int32)
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"][frame_idx] = torch.cat([inference_state["obj_ids"][frame_idx], obj_ids])
            inference_state["max_obj_id"] = max(obj_ids.tolist() + [inference_state["max_obj_id"]])
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)
            inference_state["parent_ids"][frame_idx] = torch.cat([inference_state["parent_ids"][frame_idx], parent_ids])

        elif obj_ids is None: # only in first frame
            num_cells = data["masks"].shape[0]
            obj_ids = torch.arange(num_cells, device=self.device, dtype=torch.int32) + 1
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"] = {frame_idx:obj_ids}
            inference_state["max_obj_id"] = max(obj_ids.tolist())
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)
            inference_state["parent_ids"] = {frame_idx:parent_ids}
            inference_state["lost_obj_ids"] = {frame_idx:torch.zeros(0, device=self.device, dtype=torch.int32)}
            inference_state["lost_high_res_masks"] = {}
        else:
            # Get all potential mother cells before NMS
            mother_ids = obj_ids[is_dividing]
            daughter_ids = torch.arange(inference_state["max_obj_id"]+1, inference_state["max_obj_id"]+1+len(mother_ids)*2, device=self.device, dtype=torch.int32)
            daughter_ids_list = daughter_ids.new_zeros((len(obj_ids),2), dtype=torch.int32)
            daughter_ids_list[is_dividing] = daughter_ids.reshape(-1,2)

            prev_obj_ids = obj_ids.clone()
            
            # Update obj_ids to include all potential daughters, even if they might be removed by NMS
            obj_ids = torch.cat([obj_ids[~is_dividing], daughter_ids])
            
            # Now filter based on NMS results
            lost_obj_ids = obj_ids[valid_next_frame_mask * (~keep_tokens)]
            inference_state["lost_obj_ids"][frame_idx] = lost_obj_ids
            if len(lost_obj_ids) > 0:
                lost_high_res_masks = save_masks[valid_next_frame_mask * (~keep_tokens)].flatten(0,1)
                lost_high_res_masks[:,(data["masks"] > self.mask_threshold).sum(0) > 0] = -torch.inf
                lost_high_res_masks = self.postprocess_mask(lost_high_res_masks, inference_state)
                inference_state["lost_high_res_masks"][frame_idx] = lost_high_res_masks > self.mask_threshold

            obj_ids = obj_ids[keep_tokens]

            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)

            # Update parent IDs for daughters that survived NMS
            for mother_id, pair_daughter_ids in zip(mother_ids, daughter_ids.reshape(-1,2)):
                if pair_daughter_ids[0] in obj_ids and pair_daughter_ids[1] in obj_ids:
                    mask0 = obj_ids == pair_daughter_ids[0]
                    mask1 = obj_ids == pair_daughter_ids[1]
                    parent_ids[mask0] = mother_id
                    parent_ids[mask1] = mother_id
                else:
                    # if one of the daughter cells is not in the final_obj_ids due to nms, then the other daughter cell must be the mother cell
                    dau_id = pair_daughter_ids[0] if pair_daughter_ids[0] in obj_ids else pair_daughter_ids[1]
                    obj_ids[obj_ids == dau_id] = mother_id
                    mother_ids = mother_ids[mother_ids != mother_id]
                    daughter_ids_list = daughter_ids_list.clone()
                    daughter_ids_list[torch.isin(daughter_ids_list, pair_daughter_ids)] = 0

            inference_state["obj_ids"][frame_idx] = obj_ids
            inference_state["max_obj_id"] = max(obj_ids.tolist() + [inference_state["max_obj_id"]])
            inference_state["parent_ids"][frame_idx] = parent_ids

        current_out["pred_masks_high_res"] = data["masks"]
        current_out["pred_object_score_logits"] = data["obj_scores"]
        current_out["obj_ptr"] = data["obj_ptr"]

        if not heatmap_input:
            assert current_out["pred_masks_high_res"].shape[0] == len(obj_ids)

        # Retrieve image features (only need to compute once for all objects)
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, current_out["pred_masks_high_res"].shape[0])
                        
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
            track_mask = np.zeros((inference_state["video_height"], inference_state["video_width"]), dtype=np.uint16)
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

        masks = masks[:,pad_top:pad_bottom,pad_left:pad_right]
        masks = masks.permute(1,2,0).cpu().numpy()
        masks = cv2.resize(masks, (inference_state["video_width"], inference_state["video_height"]))
        if masks.ndim == 2:
            masks = masks[None,...]
        else:
            masks = masks.transpose(2,0,1)

        return masks
    
    def get_input_points_from_heatmap(self, inference_state, frame_idx):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, 1)
                        
        heatmap_predictions = self.model.get_heatmap_predictions(current_vision_feats, feat_sizes)[0,0]
        input_points = self.extract_peak_points(heatmap_predictions)
        point_labels = torch.ones((input_points.shape[0], 1), dtype=torch.int, device=self.device)

        return input_points, point_labels
    
    def extract_peak_points(self, heatmap: torch.Tensor, max_points=20, min_dist=2, threshold=0.1):
        """
        Extract up to max_points from heatmap using local max suppression.
        
        Args:
            heatmap: (H, W) float tensor on GPU
            max_points: max number of points to return
            min_dist: radius (in pixels) for local suppression
            threshold: min confidence to consider a valid point

        Returns:
            points: (N, 2) float tensor of (x, y)
        """
        # Apply max pooling to find local maxima
        heatmap = heatmap.sigmoid()
        pooled = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=min_dist*2+1, stride=1, padding=min_dist)
        is_peak = (heatmap == pooled[0, 0]) & (heatmap > threshold)

        ys, xs = torch.nonzero(is_peak, as_tuple=True)
        if len(xs) == 0:
            return torch.empty((0, 2), device=heatmap.device)

        scores = heatmap[ys, xs]
        sorted_idx = torch.argsort(scores, descending=True)

        # Take top max_points
        xs = xs[sorted_idx][:max_points]
        ys = ys[sorted_idx][:max_points]

        points = torch.stack([xs, ys], dim=1).float()  # (N, 2)
        points = points.unsqueeze(1) # (N, 1, 2)
        assert heatmap.shape[0] == heatmap.shape[1] and self.model.image_size % heatmap.shape[0] == 0
        points = points * (self.model.image_size // heatmap.shape[0])
        return points


    def save_ctc(self, track_mask, frame_idx, inference_state):

        res_path = inference_state["res_path"]

        cell_ids_track_mask = np.unique(track_mask)
        cell_ids_track_mask = cell_ids_track_mask[cell_ids_track_mask != 0]

        cell_ids = inference_state["obj_ids"][frame_idx].cpu().numpy()

        assert sorted(cell_ids_track_mask) == sorted(cell_ids), "cell_ids_track_mask and cell_ids must be the same"

        if len(cell_ids) > 0:
            assert max(cell_ids) < 65536, "cell_id must be less than 65536"

        cv2.imwrite(str(res_path / f'mask_{frame_idx:03d}.tif'), track_mask.astype(np.uint16))

        if not self.segment:

            parent_ids = inference_state["parent_ids"][frame_idx].cpu().numpy()
            res_track = inference_state["res_track"]

            for cell_id, parent_id in zip(cell_ids, parent_ids):
                if cell_id not in res_track[:,0]:
                    res_track = np.concatenate([res_track, np.array([[cell_id, frame_idx, frame_idx, parent_id]])], axis=0)
                else:
                    assert res_track[res_track[:,0] == cell_id,2] == frame_idx-1, "cell_id must be continuous"
                    res_track[res_track[:,0] == cell_id,2] = frame_idx

            np.savetxt(res_path / 'res_track.txt',res_track,fmt='%d')

            inference_state["res_track"] = res_track

    def save_tracking_results(self, inference_state, tracking_results, alpha=0.3):
        res_path = inference_state["res_path"]

        if self.segment:
            num_colors = 1000
        else:
            num_colors = inference_state["max_obj_id"] + 1  # Add 1 to account for 0-based indexing
        colors = np.random.randint(0, 255, (num_colors, 3))
        color_stack = np.zeros((len(tracking_results), inference_state["video_height"], 
                              inference_state["video_width"], 3), dtype=np.uint8)

        for frame_idx, track_mask in enumerate(tracking_results):
            img = cv2.imread(str(inference_state["video_path"] / f't{frame_idx:03d}.tif'))

            # Create a colored overlay image
            overlay = np.zeros_like(img)
            
            cell_ids = np.unique(track_mask)
            cell_ids = cell_ids[cell_ids != 0]  # Exclude background (0)
            
            # Add colored masks for each cell
            for cell_id in cell_ids:
                mask = track_mask == cell_id
                overlay[mask] = colors[cell_id]
            
            # Blend original image with colored overlay
            color_stack[frame_idx] = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

            for cell_id in cell_ids:
                mask = track_mask == cell_id
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue
                
                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))

                cv2.putText(color_stack[frame_idx], 
                          str(cell_id),
                          (centroid_x-5, centroid_y+3),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,  # Font scale
                          (0, 0, 0),  # black color
                          1,    # Line thickness
                          cv2.LINE_AA)
                
            if not self.segment:
                parent_ids = inference_state["parent_ids"][frame_idx].cpu().numpy()
                parent_ids_unique = np.unique(parent_ids)
                parent_ids_unique = parent_ids_unique[parent_ids_unique != 0]

                for parent_id in parent_ids_unique:
                    dau_cell_ids = inference_state["obj_ids"][frame_idx][parent_ids == parent_id].cpu().numpy()

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
                                cv2.line(color_stack[frame_idx],
                                        (centroid1_x, centroid1_y),
                                        (centroid2_x, centroid2_y),
                                        (0, 0, 0),  # Black color
                                        1)  # Line thickness

            # Add frame number to top of frame
            cv2.putText(color_stack[frame_idx],
                        f"{frame_idx:03}",
                        (0, 15), # Position in top-left
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # Font scale 
                        (255, 255, 255),  # White color
                        1,    # Line thickness
                        cv2.LINE_AA)

        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mode = 'segment' if self.segment else 'track'
        out = cv2.VideoWriter(str(res_path / f'pred_{mode}_video.mp4'), 
                            fourcc, 10.0, # 10 fps
                            (inference_state["video_width"], inference_state["video_height"]))
        
        for frame in color_stack:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        out.release()



            

