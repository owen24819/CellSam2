# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_background_masks,
    get_next_point,
    sample_box_points,
)
from sam2.utils.misc import concat_points
from training.debug_viz import create_temporal_matching_visualization
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
        # whether to use variable number of correction points (0 to num_correction_pt_per_frame)
        # this makes training more robust by teaching model to track from obj_ptrs with varying refinement
        variable_num_correction_pt=False,
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
        self.variable_num_correction_pt = variable_num_correction_pt
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
        # input.masks is now a tensor with shape [T, max_objects, H, W]
        # Filter out padded entries using is_real and convert to lists per time step
        gt_masks_per_frame = {
            stage_id: input.masks[stage_id][input.is_real_masks[stage_id]].unsqueeze(1)  # [num_real_masks, 1, H_im, W_im]
            for stage_id in range(input.num_frames)
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

                    num_bkgd_pts = step_t_is_bkgd_mask.sum()

                    if num_bkgd_pts > 0:# and self.rng.random() > 0.1:
                        assert bkgd_masks.shape[1] == 1, "Needs updating for a batch size greater than 1"
                        bkgd_points = self.get_input_points_from_heatmap(input, num_bkgd_pts, bkgd_masks[0,0])
                        if bkgd_points.shape[0] > 0:
                            points[-bkgd_points.shape[0]:] = bkgd_points

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Determine number of correction points for this sample
        # Variable refinement makes the model robust to different amounts of iterative refinement
        if self.training and self.variable_num_correction_pt:
            # Randomly sample 0 to num_correction_pt_per_frame (inclusive)
            # This teaches model to track from obj_ptrs with varying levels of refinement
            num_corrections = self.rng.integers(0, self.num_correction_pt_per_frame + 1)
        else:
            num_corrections = self.num_correction_pt_per_frame
        backbone_out["num_corrections"] = num_corrections
        
        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_corrections == 0:
            # no correction clicks when variable refinement samples 0
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

        # input.metadata.unique_objects_identifier is now a tensor [T, max_objects, 3]
        # Filter out padded entries and get object IDs from first frame
        tracking_object_ids = input.metadata.unique_objects_identifier[0][input.is_real[0]][:, 1]
        memory_dict = {'mask_mem_pos_enc': None}
        all_frame_outputs = {}

        for stage_id in processing_order:
            if input.no_inputs[stage_id]:
                continue

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
                num_corrections=backbone_out["num_corrections"],
            )

            all_frame_outputs[stage_id] = current_out

        # Compute temporal matching for consecutive frame pairs
        if self.enable_temporal_aux_matcher:
            child_to_parent = self._build_child_to_parent_map(input, processing_order)
            active_frames = [t for t in processing_order if not input.no_inputs[t]]
            for i in range(len(active_frames) - 1):
                t0 = active_frames[i]
                t1 = active_frames[i + 1]
                out_t0 = all_frame_outputs.get(t0)
                out_t1 = all_frame_outputs.get(t1)
                if out_t0 is None or out_t1 is None:
                    continue
                if "key_tokens" not in out_t0 or "key_tokens" not in out_t1:
                    continue

                # Keys: post-div tokens from frame t.
                key_ids = out_t0["key_ids"]
                key_tokens = out_t0["key_tokens"]
                key_centroids = out_t0["key_centroids"]

                # Queries: post-div tokens for frame t+1 (tracking_object_ids order).
                query_valid = out_t1["query_valid_mask"]
                query_ids = out_t1["query_ids"][query_valid]

                if key_ids.numel() == 0 or query_ids.numel() == 0:
                    continue

                # Select query token mode.
                #
                # Pair (0 → 1): the key frame is the initial conditioning frame, which has
                # no prior memory (empty memory dict at t=0).  Its obj_ptr is therefore
                # already in "no-memory" mode, giving us a naturally fresh key.  We always
                # run detection queries here so the head sees consistent (no-mem key,
                # no-mem query) pairs — directly matching segment=True inference.
                #
                # Later pairs (t > 0 → t+1): keys are memory-conditioned.  We use the
                # per-frame 50/50 flag (_temporal_aux_use_conditioned_queries) to train
                # on both conditioned and detection queries, as before.
                #
                # This is done to train it for the "new_cells_only" and "segment_then_aux_track" mode
                if t0 == 0:
                    query_tokens, query_centroids = self._compute_detection_query_tokens(
                        out_t1, query_valid, input, t1, query_ids
                    )
                    query_recomputed = True  # detection path
                    if query_tokens is None or query_tokens.shape[0] == 0:
                        continue
                else:
                    # Memory-conditioned keys; 50/50 conditioned vs detection for queries.
                    if out_t1.get("_temporal_aux_use_conditioned_queries", True):
                        query_tokens = out_t1["key_tokens"][query_valid]
                        query_centroids = out_t1["key_centroids"][query_valid]
                        query_recomputed = False
                    else:
                        query_tokens, query_centroids = self._compute_detection_query_tokens(
                            out_t1, query_valid, input, t1, query_ids
                        )
                        query_recomputed = True
                    if query_tokens is None or query_tokens.shape[0] == 0:
                        continue

                # Keys built after PT for frame 0 only
                key_recomputed = t0 == 0

                # Detach tokens so loss_match gradients only update the
                # temporal head — not the SAM decoder / LoRA / backbone.
                match_logits = self.temporal_matching_head(
                    query_tokens.detach(), key_tokens.detach(),
                    query_centroids, key_centroids,
                )
                match_targets = self._build_matching_targets(
                    query_ids, key_ids, child_to_parent,
                )

                if getattr(self, "_do_temporal_debug_viz", False):
                    key_masks = out_t0.get("pred_masks")
                    query_masks = out_t1.get("pred_masks")[query_valid]
                    create_temporal_matching_visualization(
                        input=input,
                        t0=t0, t1=t1,
                        key_ids=key_ids, key_centroids=key_centroids,
                        query_ids=query_ids, query_centroids=query_centroids,
                        match_logits=match_logits, match_targets=match_targets,
                        child_to_parent=child_to_parent,
                        save_dir=getattr(self, "_temporal_debug_viz_dir", "debug_temporal_matching"),
                        step=getattr(self, "_temporal_debug_viz_sample_idx", 0),
                        key_masks=key_masks,
                        query_masks=query_masks,
                        key_recomputed=key_recomputed,
                        query_recomputed=query_recomputed,
                    )

                all_frame_outputs[t1]["temporal_match_logits"] = match_logits
                all_frame_outputs[t1]["temporal_match_targets"] = match_targets

        # turn `output_dict` into a list for loss function
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames) if not input.no_inputs[t]]
        # Remove per-frame metadata used only during matching (not needed for loss)
        _matching_scratch_keys = {
            "key_tokens", "key_centroids", "key_areas", "key_ids",
            "query_ids", "query_valid_mask",
            "_raw_vision_feats", "_high_res_features",
            "_feat_sizes", "_query_masks",
            "_temporal_aux_use_conditioned_queries",
        }
        all_frame_outputs = [
            {k: v for k, v in d.items()
             if k != "obj_ptr" and k not in _matching_scratch_keys}
            for d in all_frame_outputs
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
        num_corrections=None,  # Number of correction points (determined in prepare_prompt_inputs)
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
            
        # Get cell division information for current frame (filter out padded entries)
        is_dividing = input.cell_divides[frame_idx][input.is_real[frame_idx]]
        
        # Set default for frames_to_add_correction_pt if None
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        # Set default for num_corrections if None
        if num_corrections is None:
            num_corrections = self.num_correction_pt_per_frame
            
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
            tracking_object_ids,
            memory_dict,
            is_dividing,
            gt_masks
        )

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

        # Cache raw backbone features for the temporal matcher (used by
        # _compute_detection_query_tokens).
        if self.enable_temporal_aux_matcher:
            # Raw backbone features for detection-mode query token generation.
            current_out["_raw_vision_feats"] = current_vision_feats
            current_out["_high_res_features"] = high_res_features
            current_out["_feat_sizes"] = feat_sizes

        # Store prediction results
        self._store_prediction_results(
            current_out,
            low_res_masks,
            high_res_masks,
            ious,
            point_inputs,
            object_score_logits_dict,
            div_score_logits,
        )

        image_tensor = input.img_batch[frame_idx, :1]
        current_out["heatmap_predictions"] = self.get_heatmap_predictions(
            current_vision_feats, feat_sizes, image_tensor=image_tensor
        )[0, 0]  # assume batch size is 1
        is_used = tracking_object_ids > 0
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
        if frame_idx in frames_to_add_correction_pt and is_used.sum() > 0:
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
                is_used=is_used,
                num_corrections=num_corrections,
            )

        # Build post-division key tokens for the temporal matcher (after PT so keys match refined masks).
        # Only include foreground (id > 0); skip background to save compute and avoid useless matching.
        if self.enable_temporal_aux_matcher and current_out.get("obj_ptr") is not None:
            key_valid = tracking_object_ids > 0
            N_foreground = key_valid.sum().item()
            if N_foreground > 0:
                raw_feat = current_vision_feats[-1]
                H_feat, W_feat = feat_sizes[-1]
                C_feat = raw_feat.size(2)
                obj_ptr_fg = current_out["obj_ptr"][key_valid]
                pred_masks_fg = current_out["pred_masks"][key_valid]
                pix_feat_keys = (
                    raw_feat[:, 0, :]
                    .view(H_feat, W_feat, C_feat)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .expand(N_foreground, -1, -1, -1)
                    .contiguous()
                )
                key_tokens, key_centroids = (
                    self.temporal_matching_head.build_matching_tokens(
                        obj_ptr_fg,
                        pix_feat_keys,
                        pred_masks_fg,
                    )
                )
                current_out["key_tokens"] = key_tokens
                current_out["key_centroids"] = key_centroids
                current_out["key_ids"] = tracking_object_ids[key_valid].clone()
                current_out["_query_masks"] = current_out["pred_masks"]
                current_out["query_ids"] = tracking_object_ids.clone()
                current_out["query_valid_mask"] = key_valid  # same as tracking_object_ids > 0
                current_out["_temporal_aux_use_conditioned_queries"] = (
                    torch.rand(1, device=tracking_object_ids.device).item() < 0.5
                )
            else:
                current_out["query_ids"] = tracking_object_ids
                current_out["query_valid_mask"] = tracking_object_ids > 0

        # Adjust vision features based on token count changes
        current_vision_feats = self._adjust_vision_features(
            pix_feat.shape[0],
            current_out["pred_masks"].shape[0],
            current_vision_feats
        )

        # Update memory with new features
        if current_out["pred_masks"].shape[0] > 0:
            # Filter out padded entries
            daughter_ids_list = input.daughter_ids[frame_idx][input.is_real[frame_idx]]
            memory_dict = self._update_memory_features(
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
                daughter_ids_list,
            )

        return current_out, tracking_object_ids, memory_dict

    def _store_prediction_results(
        self,
        current_out,
        low_res_masks,
        high_res_masks,
        ious,
        point_inputs,
        object_score_logits_dict,
        div_score_logits,
    ):
        
        """Store prediction results in the output dictionary."""
        current_out["multistep_pred_masks"] = [low_res_masks]
        current_out["multistep_pred_masks_high_res"] = [high_res_masks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits_dict["pre_div"]]
        current_out["multistep_div_score_logits"] = [div_score_logits]
        current_out["post_split_object_score_logits"] = [object_score_logits_dict["post_div"]]
        
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
        # Get cell tracking mask for current frame (filter out padded entries)
        # You may want to continue to track cells that have exited FOV last frame
        # e.g. the bkgd points are tracked for 2 frames even tho segment.sum() == 0 where only cls loss would be applied
        cell_tracks_mask = input.cell_tracks_mask[frame_idx][input.is_real[frame_idx]]
        
        # Store pre-division target objects (filter out padded entries)
        pre_div_target_obj = input.target_obj_mask[frame_idx][input.is_real[frame_idx]].float()[:,None]
        current_out["pre_div_target_obj"] = [pre_div_target_obj]

        current_out["target_obj_divides"] = [is_dividing.float()[:,None]]

        # Create mask for tokens to keep after division
        post_div_target_obj = torch.cat((
            pre_div_target_obj[~is_dividing], 
            torch.ones((is_dividing.sum()*2, 1), device=cell_tracks_mask.device).float()
        ))
        current_out["post_div_target_obj"] = [post_div_target_obj]

        # Create mask for tokens to keep after division
        keep_tokens_mask = torch.cat((
            cell_tracks_mask[~is_dividing], 
            torch.ones(is_dividing.sum()*2, device=cell_tracks_mask.device).bool()
        ))
        current_out["multistep_is_point_used"] = [torch.ones_like(keep_tokens_mask).bool()]

        # Update tracking object IDs to account for cell division
        prev_tracking_object_ids = tracking_object_ids.clone()
        assert (tracking_object_ids == input.metadata.unique_objects_identifier[frame_idx][input.is_real[frame_idx]][:, 1]).all(), "Tracking object IDs do not match the input object IDs"
        
        # Get new daughter cell IDs (filter out padded entries)
        daughter_ids = input.daughter_ids[frame_idx][input.is_real[frame_idx]]
        mother_ids = tracking_object_ids[(daughter_ids > 0).any(1)]
        
        # For cell that divides into one daughter cell, swap out mother ID with daughter ID
        one_dau_mask = (daughter_ids > 0).sum(1) == 1
        tracking_object_ids[one_dau_mask] = daughter_ids[one_dau_mask,0]
        
        # For cell that divides into two daughter cells, remove mother ID and add daughter IDs at end
        two_daughter_ids = daughter_ids[(daughter_ids > 0).all(1)].flatten()
        tracking_object_ids = torch.cat((tracking_object_ids[~is_dividing], two_daughter_ids))

        # Filter out objects that are no longer tracked
        tracking_object_ids = tracking_object_ids[keep_tokens_mask]

        # Update object pointers
        obj_ptrs = obj_ptr[keep_tokens_mask]
        current_out["obj_ptr"] = obj_ptrs
        
        # Update mask predictions
        current_out["pred_masks"] = current_out["multistep_pred_masks"][0][keep_tokens_mask]
        current_out["pred_masks_high_res"] = current_out["multistep_pred_masks_high_res"][0][keep_tokens_mask]
        current_out["pred_object_score_logits"] = current_out["post_split_object_score_logits"][0][keep_tokens_mask]
        current_out["tracking_object_ids"] = tracking_object_ids  # Store for visualization
        
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


    def get_input_points_from_heatmap(self, input, num_bkgd_pts, bkgd_masks):

        img_ids = input.flat_obj_to_img_idx[0]

        (
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features_per_frame(
            input.flat_img_batch, img_ids
        )


        image_tensor = input.flat_img_batch[:1]
        heatmap_predictions = self.get_heatmap_predictions(
            current_vision_feats, feat_sizes, image_tensor=image_tensor
        )[0, 0]
        points = self.extract_peak_points(heatmap_predictions)

        # Handle edge case: no points extracted from heatmap
        if points.shape[0] == 0:
            # Return empty tensor with correct shape [0, 1, 2]
            return torch.empty((0, 1, 2), device=heatmap_predictions.device, dtype=torch.long)

        # Convert input_points to integer indices
        points = points.long()  # Shape: [212, 1, 2]

        # Get the values from bkgd_masks[0,0] at each point
        bkgd_point_values = bkgd_masks[points[:,0,1], points[:,0,0]]  # Shape: [212]

        # Get the first num_bkgd_pts indices where mask is False
        top_bkgd_pt_indices = bkgd_point_values.nonzero()[:num_bkgd_pts]

        # Handle edge case: no background points found
        if top_bkgd_pt_indices.numel() == 0:
            # Return empty tensor with correct shape [0, 1, 2]
            return torch.empty((0, 1, 2), device=heatmap_predictions.device, dtype=torch.long)

        top_bkgd_pt_indices = top_bkgd_pt_indices.squeeze(-1)
        
        # Get the corresponding points
        bkgd_points = points[top_bkgd_pt_indices]  # Shape: [num_bkgd_pts, 1, 2]

        return bkgd_points

    # ------------------------------------------------------------------
    # Temporal matching helpers
    # ------------------------------------------------------------------

    def _get_gt_masks_for_frame_in_query_order(self, input, frame_idx, query_ids):
        """Return GT masks [N_valid, 1, H, W] in the same order as query_ids (cell IDs).

        Uses the same ordering as collate: non-dividing cell masks then daughter masks.
        """
        is_real_t = input.is_real[frame_idx]
        is_real_masks_t = input.is_real_masks[frame_idx]
        gt_masks_t = input.masks[frame_idx][is_real_masks_t]  # [num_masks, H, W]
        if gt_masks_t.numel() == 0:
            return None

        cell_ids_raw = input.metadata.unique_objects_identifier[frame_idx][is_real_t][:, 1]
        cell_divides_t = input.cell_divides[frame_idx][is_real_t]
        daughter_ids_t = input.daughter_ids[frame_idx][is_real_t]

        gt_cell_ids = []
        for i in range(len(cell_ids_raw)):
            if cell_divides_t[i]:
                for d_id in daughter_ids_t[i]:
                    if d_id > 0:
                        gt_cell_ids.append(d_id.item())
            else:
                gt_cell_ids.append(cell_ids_raw[i].item())

        cell_id_to_idx = {cid: j for j, cid in enumerate(gt_cell_ids)}
        device = query_ids.device
        N_valid = query_ids.shape[0]
        H_m, W_m = gt_masks_t.shape[1], gt_masks_t.shape[2]
        valid_gt_masks = torch.zeros(
            N_valid, 1, H_m, W_m, dtype=torch.float32, device=device
        )
        for i in range(N_valid):
            qid = query_ids[i].item()
            if qid not in cell_id_to_idx:
                continue
            j = cell_id_to_idx[qid]
            valid_gt_masks[i, 0] = gt_masks_t[j].float().to(device)
        return valid_gt_masks

    def _compute_detection_query_tokens(self, out_t1, query_valid, input, t1, query_ids):
        """Build detection-mode query tokens for frame t+1.

        Uses GT mask centroids as point prompts (stable, not near touching cells).
        Runs SAM decoder on raw backbone features, then builds tokens and centroids
        from the high-res mask output.
        """
        raw_vision_feats = out_t1.get("_raw_vision_feats")
        high_res_features = out_t1.get("_high_res_features")
        feat_sizes = out_t1.get("_feat_sizes")

        if raw_vision_feats is None:
            return None, None

        valid_mask_idx = query_valid.nonzero(as_tuple=False).squeeze(1)
        if valid_mask_idx.numel() == 0:
            return None, None

        device = raw_vision_feats[0].device
        N_valid = valid_mask_idx.numel()

        # GT masks for each query_id (same order as query_ids)
        gt_masks = self._get_gt_masks_for_frame_in_query_order(input, t1, query_ids)
        if gt_masks is None or gt_masks.shape[0] != N_valid:
            return None, None

        # Centroids from GT masks in [0, 1] (same convention as build_matching_tokens)
        mask_bin = gt_masks.squeeze(1)  # [N_valid, H_m, W_m]
        area = mask_bin.sum(dim=(1, 2)).clamp(min=1)
        H_m, W_m = mask_bin.shape[1], mask_bin.shape[2]
        grid_y = torch.arange(H_m, device=device).float() / max(H_m - 1, 1)
        grid_x = torch.arange(W_m, device=device).float() / max(W_m - 1, 1)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        cx_full = (mask_bin * grid_x).sum(dim=(1, 2)) / area
        cy_full = (mask_bin * grid_y).sum(dim=(1, 2)) / area

        raw_feat = raw_vision_feats[-1]
        H_feat, W_feat = feat_sizes[-1]
        C_feat = raw_feat.size(2)
        raw_feat_bchw = (
            raw_feat[:, 0, :]
            .view(H_feat, W_feat, C_feat)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(N_valid, -1, -1, -1)
            .contiguous()
        )
        det_high_res = None
        if high_res_features is not None and self.use_high_res_features_in_sam:
            det_high_res = []
            for hr_feat in high_res_features:
                det_high_res.append(
                    hr_feat[0].unsqueeze(0).expand(N_valid, -1, -1, -1).contiguous()
                )
        img_sz = float(self.image_size)
        point_coords = torch.stack([cx_full * img_sz, cy_full * img_sz], dim=1).unsqueeze(1)
        point_labels = torch.ones(N_valid, 1, dtype=torch.int32, device=device)
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}

        is_div_query = torch.zeros(N_valid, dtype=torch.bool, device=device)
        (
            _ious, _low_res, _high_res, _obj_ptr,
            _obj_score, _div_score, _is_div,
        ) = self._forward_sam_heads(
            backbone_features=raw_feat_bchw,
            point_inputs=point_inputs,
            high_res_features=det_high_res,
            is_dividing=is_div_query,
        )

        # Use low-res masks so query tokens/centroids match key side (both low-res) for 0→1 consistency.
        tokens, centroids = self.temporal_matching_head.build_matching_tokens(
            _obj_ptr, raw_feat_bchw, _low_res,
        )
        return tokens, centroids

    def _build_child_to_parent_map(self, input, processing_order):
        """Build {daughter_id: mother_id} across all frames.

        Used for ground-truth targets (detection query D matches mother C's key) and
        for sampling priority (daughter queries are highest priority).
        """
        child_to_parent = {}
        if not hasattr(input, 'daughter_ids') or input.daughter_ids is None:
            return child_to_parent
        if not hasattr(input, 'cell_divides') or input.cell_divides is None:
            return child_to_parent
        for t in processing_order:
            if input.no_inputs[t]:
                continue
            real_mask = input.is_real[t]
            gt_ids = input.metadata.unique_objects_identifier[t][real_mask][:, 1]
            cell_divides_t = input.cell_divides[t][real_mask]
            daughter_ids_t = input.daughter_ids[t][real_mask]
            for i, (is_div, parent_id) in enumerate(zip(cell_divides_t, gt_ids)):
                if not is_div:
                    continue
                for d_id in daughter_ids_t[i]:
                    d_id_val = d_id.item()
                    if d_id_val > 0:
                        child_to_parent[d_id_val] = parent_id.item()
        return child_to_parent

    def _build_matching_targets(self, query_ids, key_ids, child_to_parent):
        """
        Args:
            query_ids: Tensor of IDs for cells in frame t+1
            key_ids: Tensor of IDs for cells in frame t
            child_to_parent: Dict mapping {daughter_id: parent_id}
        """
        device = query_ids.device
        N_k = len(key_ids)

        # Map key IDs to their position in the key_tokens tensor
        key_id_to_idx = {kid.item(): idx for idx, kid in enumerate(key_ids)}

        targets = []
        for qid in query_ids:
            qid_val = qid.item()

            # 1. Check if it's a direct track (same ID exists in previous frame)
            if qid_val in key_id_to_idx:
                targets.append(key_id_to_idx[qid_val])

            # 2. Check if it's a daughter (its parent ID exists in previous frame)
            elif qid_val in child_to_parent and child_to_parent[qid_val] in key_id_to_idx:
                parent_id = child_to_parent[qid_val]
                targets.append(key_id_to_idx[parent_id])

            # 3. New cell/No match: point to the NULL key (at index N_k)
            else:
                targets.append(N_k)

        return torch.tensor(targets, device=device, dtype=torch.long)

    def _iter_correct_pt_sampling(
        self,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        current_out,
        keep_tokens_mask,
        is_used,
        num_corrections,
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
            num_corrections: Number of correction points to sample
        
        Returns:
            Updated current_out dictionary with iterative correction results
        """
        # Filter inputs based on keep_tokens_mask
        gt_masks = gt_masks[is_used]
        high_res_features = [feat[is_used] for feat in high_res_features]
        pix_feat_with_mem = pix_feat_with_mem[is_used]
        
        point_inputs = {
            'point_coords': point_inputs['point_coords'][is_used],
            'point_labels': point_inputs['point_labels'][is_used],
        }
        
        # Get initial masks from the first prediction step
        low_res_masks = current_out["multistep_pred_masks"][0][is_used]
        high_res_masks = current_out["multistep_pred_masks_high_res"][0][is_used]
        is_dividing = torch.zeros(low_res_masks.shape[0], dtype=torch.bool)
        
        assert gt_masks is not None, "Ground truth masks required for correction point sampling"
        
        # Iteratively add correction points
        for _ in range(num_corrections):
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
                object_score_logits_dict,
                div_score_logits,
                is_dividing,
            ) = sam_outputs
            
            # Store results for this correction step
            current_out["multistep_pred_masks"].append(low_res_masks)
            current_out["multistep_pred_masks_high_res"].append(high_res_masks)
            current_out["multistep_pred_ious"].append(ious)
            current_out["multistep_point_inputs"].append(point_inputs)
            current_out["multistep_object_score_logits"].append(object_score_logits_dict["pre_div"])
            current_out["multistep_div_score_logits"].append(div_score_logits)
            current_out["post_split_object_score_logits"].append(object_score_logits_dict["post_div"])
            current_out["multistep_is_point_used"].append(is_used)
            
            current_out["pre_div_target_obj"].append(current_out["pre_div_target_obj"][0].clone()[is_used])
            current_out["post_div_target_obj"].append(current_out["post_div_target_obj"][0].clone()[is_used])
            current_out["target_obj_divides"].append(current_out["target_obj_divides"][0].clone()[is_used])
        
        # Update final predictions for memory encoder
        current_out["obj_ptr"][is_used[keep_tokens_mask]] = obj_ptr[keep_tokens_mask[is_used]]
        current_out["pred_masks"][is_used[keep_tokens_mask]] = low_res_masks[keep_tokens_mask[is_used]]
        current_out["pred_masks_high_res"][is_used[keep_tokens_mask]] = high_res_masks[keep_tokens_mask[is_used]]
        
        return current_out