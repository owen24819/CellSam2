# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # Whether to predict if there is a cell division in the frame
        pred_div_scores: bool = False,
        # Whether to use an MLP to predict cell division scores
        pred_div_scores_mlp: bool = False,
        pred_iou_thresh: float = 0.7,
        obj_score_thresh: float = 0.5,
        div_obj_score_thresh: float = 0.5,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.pred_div_scores = pred_div_scores
        self.pred_div_scores_mlp = pred_div_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        
        self.pred_iou_thresh = pred_iou_thresh
        self.obj_score_thresh = obj_score_thresh
        self.div_obj_score_thresh = div_obj_score_thresh

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            pred_div_scores=self.pred_div_scores,
            pred_div_scores_mlp=self.pred_div_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            pred_iou_thresh=self.pred_iou_thresh,
            obj_score_thresh=self.obj_score_thresh,
            div_obj_score_thresh=self.div_obj_score_thresh,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

        self.heatmap_predictor = torch.nn.Sequential(
            torch.nn.Conv2d(self.hidden_dim // 8 * 3, self.hidden_dim // 8, kernel_size=3, padding=1),  # Local context
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_dim // 8, self.hidden_dim // 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_dim // 8, 1, kernel_size=1),  # Compress to heatmap
        )

        self.feature_dim_reducers = torch.nn.ModuleList([
            torch.nn.Conv2d(self.hidden_dim // 8, self.hidden_dim // 8, kernel_size=1),
            torch.nn.Conv2d(self.hidden_dim // 4, self.hidden_dim // 8, kernel_size=1), 
            torch.nn.Conv2d(self.hidden_dim, self.hidden_dim // 8, kernel_size=1)
        ])

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        is_dividing=None,
        gt_masks=None,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - is_dividing: Optional tensor indicating which cells are dividing
        - gt_masks: Optional ground truth masks for training

        Outputs:
        - ious: [B, 1] shape, the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape.
        - high_res_masks: [B, 1, H*16, W*16] shape.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask.
        - object_score_logits: [B, 1] shape, the object score logits for the output mask.
        - div_score_logits: [B, 1] shape, the cell division score logits for the output mask.
        - post_split_object_score_logits: Score logits for objects after division.
        - is_dividing: Tensor indicating which cells are dividing pre division.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # Process point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # Create empty point prompt with padding label (-1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # Process mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        
        # Generate masks through the decoder
        (
            low_res_masks,
            ious,
            sam_output_tokens,
            object_score_logits_dict,
            div_score_logits,
            is_dividing,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
            is_dividing=is_dividing,
            gt_masks=gt_masks,
        )

        num_non_dividing_cells = (~is_dividing).sum()

        # Apply object score thresholding for non-dividing cells
        if self.pred_obj_scores and num_non_dividing_cells > 0:
            is_obj_appearing = object_score_logits_dict["post_div"] > 0
            
            # Apply hard thresholding to low_res_masks based on object scores
            # Set masks to NO_OBJ_SCORE where object is not appearing
            low_res_masks[:num_non_dividing_cells] = torch.where(
                is_obj_appearing[:num_non_dividing_cells, None, None],
                low_res_masks[:num_non_dividing_cells],
                NO_OBJ_SCORE,
            )

        # Ensure masks are float32 for interpolation compatibility
        low_res_masks = low_res_masks.float()
        
        # Upsample masks to high resolution
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Extract and process object pointers
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        
        # Apply object score conditioning to object pointers
        if self.pred_obj_scores and num_non_dividing_cells > 0:
            # Calculate object appearance factor (soft or hard threshold)
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits_dict["post_div"][:num_non_dividing_cells].sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing[:num_non_dividing_cells].float()

            # Apply fixed no-object pointer if configured
            if self.fixed_no_obj_ptr:
                obj_ptr[:num_non_dividing_cells] = lambda_is_obj_appearing * obj_ptr[:num_non_dividing_cells]
                
            # Mix in no-object pointer based on object appearance factor
            obj_ptr[:num_non_dividing_cells] = obj_ptr[:num_non_dividing_cells] + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits_dict,
            div_score_logits,
            is_dividing,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.to(torch.float32)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        num_frames,
        tracking_object_ids,
        memory_dict,
    ):
        """
        Fuse the current frame's visual feature map with previous memory.
        
        Args:
            frame_idx: Index of the current frame
            is_init_cond_frame: Whether this is an initial conditioning frame
            current_vision_feats: List of feature maps from the vision encoder
            current_vision_pos_embeds: List of positional embeddings
            feat_sizes: Sizes of the feature maps
            output_dict: Dictionary containing outputs from previous frames
            num_frames: Total number of frames
            tracking_object_ids: IDs of objects being tracked
            memory_dict: Dictionary containing memory features for each object
            track_in_reverse: Whether tracking is in reverse time order
            
        Returns:
            Tensor of shape [B, C, H, W] containing the fused features
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        
        # Skip memory fusion for image-only mode
        if self.num_maskmem == 0 or (tracking_object_ids is not None and len(tracking_object_ids) == 0):
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        # Handle initial conditioning frames differently
        if is_init_cond_frame:
            if self.directly_add_no_mem_embed:
                # Directly add no-memory embedding to features
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use dummy tokens for initial frames
            memory = self.no_mem_embed.expand(1, B, self.mem_dim)
            memory_pos_embed = self.no_mem_pos_enc.expand(1, B, self.mem_dim)
            num_obj_ptr_tokens = 0
        else:
            # Process memory for non-initial frames
            memory, memory_pos_embed, num_obj_ptr_tokens = self._prepare_memory_for_attention(
                B, C, H, W, device, tracking_object_ids, memory_dict, num_frames
            )

        # Apply memory attention to fuse features
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        
        # Reshape output from (HW)BC to BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _prepare_memory_for_attention(self, B, C, H, W, device, tracking_object_ids, memory_dict, num_frames):
        """
        Helper method to prepare memory tensors for attention.
        
        Returns:
            Tuple of (memory, memory_pos_embed, num_obj_ptr_tokens)
        """
        # Initialize memory tensors
        memory = torch.zeros(self.num_maskmem, B, self.mem_dim, H, W, device=device)
        N = 0  # Actual number of memory frames to use
        
        # Initialize object pointer tensors if needed
        if self.use_obj_ptrs_in_encoder:
            if self.mem_dim < C:
                obj_ptrs_mem = torch.zeros(
                    self.num_maskmem, B, C // self.mem_dim, self.mem_dim, device=device
                )
            else:
                raise NotImplementedError("Memory dimension is not supported for obj ptrs")
        
        # Process each tracked object
        for idx, object_id in enumerate(tracking_object_ids):
            if not isinstance(object_id, int):
                object_id = object_id.item()
            
            # Get memory features for this object if available
            if object_id in memory_dict:
                mask_mem_features = memory_dict[object_id]["mask_mem_features"]
                num_mem_frames = min(mask_mem_features.shape[0], self.num_maskmem)
                memory[:num_mem_frames, idx] = mask_mem_features[-num_mem_frames:]
                N = max(N, num_mem_frames)
            else:
                num_mem_frames = 0
            
            # Fill remaining memory slots with no-object embedding
            if self.no_obj_embed_spatial is not None:
                memory[num_mem_frames:, idx] = self.no_obj_embed_spatial[0, :, None, None].expand(
                    self.num_maskmem - num_mem_frames, self.mem_dim, H, W
                )
            
            # Process object pointers if enabled
            if self.use_obj_ptrs_in_encoder:
                if object_id in memory_dict:
                    obj_ptrs = memory_dict[object_id]["obj_ptr"][-num_mem_frames:]
                    # Split pointer into tokens for mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(-1, C // self.mem_dim, self.mem_dim)
                    obj_ptrs_mem[:num_mem_frames, idx] = obj_ptrs
                
                # Fill remaining slots with no-object pointer
                obj_ptrs_mem[num_mem_frames:, idx] = self.no_obj_ptr[None].reshape(
                    1, C // self.mem_dim, self.mem_dim
                ).expand(self.num_maskmem - num_mem_frames, C // self.mem_dim, self.mem_dim)
        
        # Trim to actual number of memory frames
        memory = memory[:N]
        
        # Reshape memory for attention: [N, B, C, H, W] -> [(N*H*W), B, C]
        memory = memory.flatten(3).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, B, self.mem_dim)
        
        # Process object pointers if enabled
        num_obj_ptr_tokens = 0
        if self.use_obj_ptrs_in_encoder:
            obj_ptrs_mem = obj_ptrs_mem[:N]
            obj_ptrs_mem = obj_ptrs_mem.permute(0, 2, 1, 3)
            obj_ptrs_mem = obj_ptrs_mem.reshape(-1, B, self.mem_dim)
            memory = torch.cat((memory, obj_ptrs_mem), dim=0)
            num_obj_ptr_tokens = obj_ptrs_mem.shape[0]
        
        # Prepare positional embeddings
        memory_pos_embed = memory_dict["mask_mem_pos_enc"]
        memory_pos_embed = memory_pos_embed[:1].flatten(2).permute(2, 0, 1)
        memory_pos_embed = memory_pos_embed[None].expand(N, H*W, B, self.mem_dim)
        
        # Add temporal positional encoding
        t_pos_indices = torch.arange(N, device=device)
        memory_pos_embed = memory_pos_embed + self.maskmem_tpos_enc[t_pos_indices].expand(N, H*W, B, self.mem_dim)
        memory_pos_embed = memory_pos_embed.reshape(-1, B, self.mem_dim)
        
        # Add object pointer positional embeddings if needed
        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            
            if self.add_tpos_enc_to_obj_ptrs:
                t_diff_max = max(max_obj_ptrs_in_encoder - 1, 1)  # Avoid division by zero
                tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                obj_pos = get_1d_sine_pe(t_pos_indices / t_diff_max, dim=tpos_dim)
                obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
            else:
                obj_pos = memory_pos_embed.new_zeros(C // self.mem_dim, B, self.mem_dim)
            
            memory_pos_embed = torch.cat((memory_pos_embed, obj_pos), dim=0)
        
        return memory, memory_pos_embed, num_obj_ptr_tokens

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
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
        is_dividing=None,
        gt_masks=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                num_frames=num_frames,
                tracking_object_ids=tracking_object_ids,
                memory_dict=memory_dict,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits

            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                is_dividing=is_dividing,
                gt_masks=gt_masks,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        current_out,
    ):
        """
        Encode the current frame's prediction into a memory feature.
        
        Args:
            current_vision_feats: Image features from the backbone
            feat_sizes: Sizes of the feature maps
            point_inputs: Point prompts if any
            run_mem_encoder: Whether to run the memory encoder
            current_out: Dictionary containing current outputs including masks
            
        Returns:
            Tuple of (maskmem_features, maskmem_pos_enc) or (None, None)
        """
        if not run_mem_encoder or self.num_maskmem <= 0:
            return None, None
        
        high_res_masks = current_out["pred_masks_high_res"]
        object_score_logits = current_out["pred_object_score_logits"]

        if high_res_masks.ndim == 3:
            high_res_masks = high_res_masks[:,None]

        if object_score_logits.ndim == 1:
            object_score_logits = object_score_logits[:,None]
        
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=(point_inputs is not None),
        )
        
        # Store the memory features in the output dictionary
        current_out["maskmem_features"] = maskmem_features
        current_out["maskmem_pos_enc"] = maskmem_pos_enc
        
        return maskmem_features, maskmem_pos_enc

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        maskmem_features, maskmem_pos_enc = self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            current_out,
        )

        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

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
        daughter_ids_list,
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
                daughter_ids = daughter_ids_list[mother_id_index]
                
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

        return memory_dict
    
    def get_heatmap_predictions(self, current_vision_feats, feat_sizes):
        """
        Generate heatmap predictions from multi-scale vision features.
        
        Args:
            current_vision_feats (List[torch.Tensor]): List of feature maps at different scales
            feat_sizes (List[Tuple[int, int]]): Original spatial dimensions for each feature map
        
        Returns:
            torch.Tensor: Predicted heatmap of shape (B, 1, H, H) where H = image_size // 4
        """
        # Define target size for all feature maps
        heatmap_size = self.image_size // 4
        
        # Process each feature map
        heatmap_vision_feats = []
        for idx, vision_feat in enumerate(current_vision_feats):
            # Reshape and reduce channel dimension
            feat = vision_feat[:, 0].permute(1, 0).reshape(1, vision_feat.shape[-1], feat_sizes[idx][0], feat_sizes[idx][1])
            # feat = vision_feat[:, :1].reshape(1,-1,feat_sizes[idx][0], feat_sizes[idx][1])
            feat = self.feature_dim_reducers[idx](feat)
            
            # Resize to target heatmap size
            feat = F.interpolate(
                feat,
                size=(heatmap_size, heatmap_size),
                mode='bilinear',
                align_corners=False
            )
            heatmap_vision_feats.append(feat)

        # Concatenate features along channel dimension
        fused_features = torch.cat(heatmap_vision_feats, dim=1)
        
        # Generate final heatmap prediction
        heatmap = self.heatmap_predictor(fused_features)
        
        return heatmap
        
    
