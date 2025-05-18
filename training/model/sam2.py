# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_next_point, sample_box_points, get_background_masks

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:

                step_t_is_bkgd_mask, bkgd_masks = get_background_masks(input, t)

                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input and step_t_is_bkgd_mask.sum() == 0: # Only sample box points if there are no bkgd points
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method="uniform" if self.training else self.pt_sampling_for_eval,
                        is_bkgd_mask=step_t_is_bkgd_mask,
                        bkgd_mask=bkgd_masks,
                        )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        tracking_object_ids = input.metadata.unique_objects_identifier[0][:,1]
        memory_dict = {}
        all_frame_outputs = {}

        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out, tracking_object_ids, memory_dict = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                num_frames=num_frames,
                input=input,
                tracking_object_ids=tracking_object_ids,
                memory_dict=memory_dict,
            )

            all_frame_outputs[stage_id] = current_out

        # turn `output_dict` into a list for loss function
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        num_frames,
        input,
        tracking_object_ids,
        memory_dict,
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        """
        Process a single frame in the tracking sequence.
        
        This method handles:
        1. Running the SAM model on the current frame
        2. Managing cell division events
        3. Updating object tracking IDs
        4. Storing memory features for temporal tracking
        5. Applying iterative correction points only on the first frame
        """
        # Initialize memory dict if first frame
        if "frame_idx" not in memory_dict:
            memory_dict["frame_idx"] = []
            
        # Get cell division information for current frame
        is_dividing = input.cell_divides[frame_idx]
        
        # Set default for frames_to_add_correction_pt if None
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
            
        # Run the core tracking step
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            num_frames,
            prev_sam_mask_logits,
            is_dividing,
            tracking_object_ids,
            memory_dict,
            gt_masks
        )

        # Unpack SAM outputs
        (
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            div_score_logits,
            post_split_object_score_logits,
            is_dividing,
        ) = sam_outputs

        # Store prediction results
        self._store_prediction_results(
            current_out,
            low_res_masks,
            high_res_masks,
            ious,
            point_inputs,
            object_score_logits,
            div_score_logits,
            post_split_object_score_logits
        )

        # Handle cell tracking and division
        keep_tokens_mask, tracking_object_ids, mother_ids, prev_tracking_object_ids = self._handle_cell_tracking(
            current_out,
            input,
            frame_idx,
            is_dividing,
            tracking_object_ids,
            obj_ptr
        )

        # Apply iterative correction points if needed
        if frame_idx in frames_to_add_correction_pt and keep_tokens_mask.sum() > 0:
            assert frame_idx == 0 and is_dividing.sum() == 0
            # Only add points to first frame
            # Maybe adapt this for other frames but will need to handle dividing cells
            current_out = self._iter_correct_pt_sampling(
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                current_out,
                keep_tokens_mask,
            )

        # Adjust vision features based on token count changes
        current_vision_feats = self._adjust_vision_features(
            pix_feat.shape[0],
            current_out["pred_masks"].shape[0],
            current_vision_feats
        )

        # Update memory with new features
        if current_out["pred_masks"].shape[0] > 0:
            self._update_memory_features(
                current_vision_feats,
                feat_sizes,
                point_inputs,
                run_mem_encoder,
                current_out,
                memory_dict,
                tracking_object_ids,
                frame_idx,
                mother_ids,
                prev_tracking_object_ids,
                input
            )

        return current_out, tracking_object_ids, memory_dict

    def _store_prediction_results(
        self,
        current_out,
        low_res_masks,
        high_res_masks,
        ious,
        point_inputs,
        object_score_logits,
        div_score_logits,
        post_split_object_score_logits
    ):
        """Store prediction results in the output dictionary."""
        current_out["multistep_pred_masks"] = [low_res_masks]
        current_out["multistep_pred_masks_high_res"] = [high_res_masks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["multistep_div_score_logits"] = [div_score_logits]
        current_out["post_split_object_score_logits"] = [post_split_object_score_logits]
        
    def _handle_cell_tracking(
        self,
        current_out,
        input,
        frame_idx,
        is_dividing,
        tracking_object_ids,
        obj_ptr
    ):
        """Handle cell tracking and division events."""
        # Get cell tracking mask for current frame
        cell_tracks_mask = input.cell_tracks_mask[frame_idx]
        
        # Store pre-division target objects
        pre_div_target_obj = cell_tracks_mask.float()[:,None]
        current_out["pre_div_target_obj"] = [pre_div_target_obj]
        
        # Create mask for tokens to keep after division
        keep_tokens_mask = torch.cat((
            cell_tracks_mask[~is_dividing], 
            torch.ones(is_dividing.sum()*2, device=cell_tracks_mask.device).bool()
        ))
        current_out["post_div_target_obj"] = [keep_tokens_mask.float()[:,None]]
        current_out["multistep_is_point_used"] = [torch.ones_like(keep_tokens_mask).bool()]

        # Update tracking object IDs to account for cell division
        prev_tracking_object_ids = tracking_object_ids.clone()
        mother_ids = tracking_object_ids[is_dividing]
        
        # Get new daughter cell IDs
        new_daughter_ids = input.daughter_ids[frame_idx].flatten()
        new_daughter_ids = new_daughter_ids[new_daughter_ids > 0]
        
        # Update tracking object IDs - swap out mother ID with daughter IDs
        tracking_object_ids = torch.cat((tracking_object_ids[~is_dividing], new_daughter_ids))
        
        # Filter out objects that are no longer tracked
        exit_object_ids = tracking_object_ids[~keep_tokens_mask]
        tracking_object_ids = tracking_object_ids[keep_tokens_mask]
        
        # if frame_idx % 10 == 0:  # Reduce logging frequency
        #     logging.debug(f'Frame: {frame_idx} | Tracking IDs: {tracking_object_ids} | Exit IDs: {exit_object_ids}')
        
        # Update object pointers
        obj_ptrs = obj_ptr[keep_tokens_mask]
        current_out["obj_ptr"] = obj_ptrs
        
        # Update mask predictions
        current_out["pred_masks"] = current_out["multistep_pred_masks"][0][keep_tokens_mask]
        current_out["pred_masks_high_res"] = current_out["multistep_pred_masks_high_res"][0][keep_tokens_mask]
        current_out["pred_object_score_logits"] = current_out["post_split_object_score_logits"][0][keep_tokens_mask]
        
        return keep_tokens_mask, tracking_object_ids, mother_ids, prev_tracking_object_ids
        
    def _adjust_vision_features(self, prev_num_tokens, cur_num_tokens, current_vision_feats):
        """Adjust vision features based on token count changes."""
        if prev_num_tokens > cur_num_tokens:
            # Reduce feature dimensions if tokens were removed
            return [feat[:, :cur_num_tokens] for feat in current_vision_feats]
        elif prev_num_tokens < cur_num_tokens:
            # Expand feature dimensions if tokens were added (e.g., cell division)
            return [
                torch.cat((
                    feat, 
                    feat[:, :1].repeat(1, cur_num_tokens - prev_num_tokens, 1)
                ), dim=1) 
                for feat in current_vision_feats
            ]
        return current_vision_feats
        
    def _update_memory_features(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        current_out,
        memory_dict,
        tracking_object_ids,
        frame_idx,
        mother_ids,
        prev_tracking_object_ids,
        input
    ):
        """Update memory features for temporal tracking."""
        # Encode current frame predictions into memory features
        maskmem_features, maskmem_pos_enc = self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            current_out,
        )
        
        if maskmem_features is None or maskmem_pos_enc is None:
            return
            
        # Store position encoding
        memory_dict["mask_mem_pos_enc"] = maskmem_pos_enc[0]
        
        # Update memory features for each tracked object
        for i, object_id in enumerate(tracking_object_ids):
            obj_id = object_id.item()
            
            if obj_id not in memory_dict:
                # Initialize memory for new object
                memory_dict[obj_id] = {
                    "mask_mem_features": maskmem_features[i:i+1], 
                    "obj_ptr": current_out["obj_ptr"][i:i+1], 
                    "frame_idx": [frame_idx]
                }
            else:
                # Update memory for existing object
                memory_dict[obj_id]["mask_mem_features"] = torch.cat(
                    (memory_dict[obj_id]["mask_mem_features"], maskmem_features[i:i+1]), 
                    dim=0
                )
                memory_dict[obj_id]["obj_ptr"] = torch.cat(
                    (memory_dict[obj_id]["obj_ptr"], current_out["obj_ptr"][i:i+1]), 
                    dim=0
                )
                memory_dict[obj_id]["frame_idx"].append(frame_idx)
        
        # Handle memory inheritance for daughter cells
        for mother_id in mother_ids:
            try:
                # Find mother cell index
                mother_id_index = torch.where(prev_tracking_object_ids == mother_id)[0].item()
                daughter_ids = input.daughter_ids[frame_idx][mother_id_index]
                
                # Verify mother cell has memory from previous frame
                mother_id_item = mother_id.item()
                if mother_id_item not in memory_dict:
                    logging.warning(f"Mother ID {mother_id_item} not found in memory dict")
                    continue
                    
                if memory_dict[mother_id_item]["frame_idx"][-1] != frame_idx-1:
                    logging.warning(f"Mother ID {mother_id_item} last frame is not {frame_idx-1}")
                    continue
                
                # Transfer mother's memory to daughters
                for daughter_id in daughter_ids:
                    if daughter_id.item() == 0:  # Skip invalid daughter IDs
                        continue
                        
                    daughter_id_item = daughter_id.item()
                    if daughter_id_item not in memory_dict:
                        logging.warning(f"Daughter ID {daughter_id_item} not found in memory dict")
                        continue
                        
                    # Prepend mother's last memory to daughter's memory
                    memory_dict[daughter_id_item]["mask_mem_features"] = torch.cat(
                        (memory_dict[mother_id_item]["mask_mem_features"][-1:], 
                         memory_dict[daughter_id_item]["mask_mem_features"]), 
                        dim=0
                    )
                    memory_dict[daughter_id_item]["obj_ptr"] = torch.cat(
                        (memory_dict[mother_id_item]["obj_ptr"][-1:], 
                         memory_dict[daughter_id_item]["obj_ptr"]), 
                        dim=0
                    )
                    memory_dict[daughter_id_item]["frame_idx"].insert(0, frame_idx-1)
            except Exception as e:
                logging.error(f"Error handling memory for mother ID {mother_id.item()}: {str(e)}")

    def _iter_correct_pt_sampling(
        self,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        current_out,
        keep_tokens_mask,
    ):
        """
        Iteratively sample correction points to improve mask predictions.
        
        Args:
            point_inputs: Dictionary containing initial point coordinates and labels
            gt_masks: Ground truth masks for evaluation
            high_res_features: High resolution features from the image encoder
            pix_feat_with_mem: Pixel features with memory
            current_out: Current output dictionary to update
            keep_tokens_mask: Boolean mask indicating which tokens to keep
        
        Returns:
            Updated current_out dictionary with iterative correction results
        """
        # Filter inputs based on keep_tokens_mask
        gt_masks = gt_masks[keep_tokens_mask]
        high_res_features = [feat[keep_tokens_mask] for feat in high_res_features]
        pix_feat_with_mem = pix_feat_with_mem[keep_tokens_mask]
        
        point_inputs = {
            'point_coords': point_inputs['point_coords'][keep_tokens_mask],
            'point_labels': point_inputs['point_labels'][keep_tokens_mask],
        }
        
        # Get initial masks from the first prediction step
        low_res_masks = current_out["multistep_pred_masks"][0][keep_tokens_mask]
        high_res_masks = current_out["multistep_pred_masks_high_res"][0][keep_tokens_mask]
        is_dividing = torch.zeros(low_res_masks.shape[0], dtype=torch.bool)
        
        assert gt_masks is not None, "Ground truth masks required for correction point sampling"
        
        # Iteratively add correction points
        for _ in range(self.num_correction_pt_per_frame):
            # Determine whether to sample from GT or error regions
            sample_from_gt = False
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = self.rng.random() < self.prob_to_sample_from_gt_for_train
                
            # If sampling from GT, don't use prediction for point selection
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            
            # Sample a new correction point
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            
            # Add the new point to existing points
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            
            # Use previous mask prediction as input for the next step
            mask_inputs = low_res_masks
            
            # Forward through SAM heads (with optional activation checkpointing)
            if self.use_act_ckpt_iterative_pt_sampling:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    use_reentrant=False,
                    is_dividing=is_dividing,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    is_dividing=is_dividing,
                )
                
            # Unpack SAM outputs
            (
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                div_score_logits,
                post_split_object_score_logits,
                is_dividing,
            ) = sam_outputs
            
            # Store results for this correction step
            current_out["multistep_pred_masks"].append(low_res_masks)
            current_out["multistep_pred_masks_high_res"].append(high_res_masks)
            current_out["multistep_pred_ious"].append(ious)
            current_out["multistep_point_inputs"].append(point_inputs)
            current_out["multistep_object_score_logits"].append(object_score_logits)
            current_out["multistep_div_score_logits"].append(div_score_logits)
            current_out["post_split_object_score_logits"].append(post_split_object_score_logits)
            current_out["multistep_is_point_used"].append(keep_tokens_mask)
            
            # No divisions in the first frame
            current_out["pre_div_target_obj"].append(torch.ones_like(div_score_logits).float())
            current_out["post_div_target_obj"].append(torch.ones_like(div_score_logits).float())
        
        # Update final predictions for memory encoder
        current_out["obj_ptr"] = obj_ptr
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        
        return current_out