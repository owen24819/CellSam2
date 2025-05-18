# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import LayerNorm2d, MLP, compute_iou


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        pred_div_scores: bool = False,
        pred_div_scores_mlp: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)

        self.pred_div_scores = pred_div_scores
        if self.pred_div_scores:
            self.div_score_token = nn.Embedding(1, transformer_dim)

        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        if self.pred_div_scores:
            self.pred_div_score_head = nn.Linear(transformer_dim, 1)
            if pred_div_scores_mlp:
                self.pred_div_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        is_dividing: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          repeat_image (bool): whether to repeat the image embeddings for each prompt
          high_res_features (Optional[List[torch.Tensor]]): optional high resolution features
          is_dividing (Optional[torch.Tensor]): optional tensor indicating dividing cells for training
          gt_masks (Optional[torch.Tensor]): ground truth masks for training

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
          torch.Tensor: batched object score logits
          torch.Tensor: batched division score logits
          torch.Tensor: batched post-split object score logits
          torch.Tensor: batched is_dividing
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, div_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Determine which cells are dividing
        if is_dividing is None:
            is_dividing = (div_score_logits > 0) & (object_score_logits > 0)
        
        # Ensure is_dividing is a flat boolean tensor
        is_dividing = is_dividing.view(-1)
        
        # Create masks for dividing and non-dividing cells
        div_mask = is_dividing  # [B] bool
        no_div_mask = ~div_mask

        # Handle non-dividing cells (single mask per cell)
        pred_masks = masks[no_div_mask, 0:1]  # Keep dim for proper shape
        pred_ious = iou_pred[no_div_mask, 0:1]
        pred_tokens = mask_tokens_out[no_div_mask, 0:1]

        # Process dividing cells if any exist
        if div_mask.sum() > 0:
            # During training with GT masks, match daughter masks to ground truth
            if self.training and gt_masks is not None:
                pred_div_masks, pred_div_ious, pred_div_tokens = self._match_daughter_masks_to_gt(
                    gt_masks, masks, iou_pred, mask_tokens_out, div_mask
                )
            else:
                # For inference or when GT masks aren't provided
                # Extract masks 1 and 2 for dividing cells and reshape
                pred_div_masks = masks[div_mask][:, 1:3]  # [N, 2, H, W]
                pred_div_ious = iou_pred[div_mask][:, 1:3]  # [N, 2]
                pred_div_tokens = mask_tokens_out[div_mask][:, 1:3]  # [N, 2, C]
                
                # Reshape to have each mask as a separate item in batch
                pred_div_masks = pred_div_masks.flatten(0, 1).unsqueeze(1)  # [N*2, 1, H, W]
                pred_div_ious = pred_div_ious.flatten(0, 1).unsqueeze(1)  # [N*2, 1]
                pred_div_tokens = pred_div_tokens.flatten(0, 1).unsqueeze(1)  # [N*2, 1, C]

            # Combine results from non-dividing and dividing cells
            pred_masks = torch.cat([pred_masks, pred_div_masks], dim=0)
            pred_ious = torch.cat([pred_ious, pred_div_ious], dim=0)
            pred_tokens = torch.cat([pred_tokens, pred_div_tokens], dim=0)

        # Update object_score_logits to match the new output structure
        post_split_object_score_logits = None
        if object_score_logits is not None:
            # For non-dividing cells, keep single score
            pred_scores = object_score_logits[no_div_mask]
            
            # Handle dividing cells if any exist
            if div_mask.any():
                # For dividing cells, duplicate scores for both daughter cells
                pred_div_scores = object_score_logits[div_mask].repeat_interleave(2, dim=0)
                # Combine scores
                post_split_object_score_logits = torch.cat([pred_scores, pred_div_scores], dim=0)
            else:
                # If no dividing cells, just use the non-dividing scores
                post_split_object_score_logits = pred_scores

        # Return all outputs
        return pred_masks, pred_ious, pred_tokens, object_score_logits, div_score_logits, post_split_object_score_logits, is_dividing

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )

        if self.pred_div_scores:
            output_tokens = torch.cat(
                [output_tokens, self.div_score_token.weight], dim=0
            )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        if self.pred_div_scores:
            div_score_logits = self.pred_div_score_head(hs[:, -1, :])
        else:
            div_score_logits = -float('inf') * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits, div_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out

    def _match_daughter_masks_to_gt(self, gt_masks, masks, iou_pred, mask_tokens_out, div_mask):
        """
        Match predicted daughter cell masks to ground truth masks by computing IoUs
        and reordering them to maximize the match.
        
        For dividing cells, this ensures the predicted daughter masks are correctly
        aligned with their corresponding ground truth masks by comparing IoUs in
        both possible orderings and selecting the ordering with the highest total IoU.
        
        Args:
            gt_masks: Ground truth masks
            masks: Predicted masks
            iou_pred: Predicted IoU scores
            mask_tokens_out: Mask tokens output from transformer
            div_mask: Boolean mask indicating which cells are dividing
            
        Returns:
            Tuple of reordered masks, IoU predictions, and mask tokens
        """
        
        assert self.training
        pred_div_masks = masks[div_mask][:, 1:3]
        pred_div_masks_sigmoid = pred_div_masks.sigmoid()  # Take masks 1 and 2
        pred_div_ious = iou_pred[div_mask][:, 1:3]
        pred_div_tokens = mask_tokens_out[div_mask][:, 1:3]

        # Daughter masks are alwaays added last
        gts = gt_masks[-div_mask.sum()*2:].reshape(-1, 2, *gt_masks.shape[-2:])      # [N, 2, H, W]
        # Resize GT masks to match prediction size
        gts = F.interpolate(
            gts.flatten(0, 1).unsqueeze(1).float(),  # [N*2, 1, H, W]
            size=pred_div_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).reshape(-1, 2, *pred_div_masks.shape[-2:])  # [N, 2, h, w]
        
        # First ordering: pred[0] with gt[0], pred[1] with gt[1]
        iou_00 = compute_iou(pred_div_masks_sigmoid[:,0], gts[:,0])
        iou_11 = compute_iou(pred_div_masks_sigmoid[:,1], gts[:,1])
        sum_iou_01 = iou_00 + iou_11

        # Swapped ordering: pred[0] with gt[1], pred[1] with gt[0]
        iou_01 = compute_iou(pred_div_masks_sigmoid[:,0], gts[:,1])
        iou_10 = compute_iou(pred_div_masks_sigmoid[:,1], gts[:,0])
        sum_iou_10 = iou_01 + iou_10

        # Choose better match
        swap = sum_iou_10 > sum_iou_01  # [N] bool

        # Create index tensor [N, 2] where each row is [0,1] or [1,0]
        order = torch.stack([
            torch.where(swap, torch.tensor(1, device=pred_div_masks.device), torch.tensor(0, device=pred_div_masks.device)),
            torch.where(swap, torch.tensor(0, device=pred_div_masks.device), torch.tensor(1, device=pred_div_masks.device))
        ], dim=1)  # [N, 2]

        # Reorder predictions accordingly
        pred_div_masks = torch.gather(pred_div_masks, dim=1, index=order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, pred_div_masks.size(2), pred_div_masks.size(3))).flatten(1).reshape(-1, 1, *pred_div_masks.shape[-2:])
        pred_div_ious = torch.gather(pred_div_ious, dim=1, index=order).flatten(1).reshape(-1, 1)
        pred_div_tokens = torch.gather(pred_div_tokens, dim=1, index=order.unsqueeze(-1).expand(-1, -1, pred_div_tokens.size(2))).flatten(1).reshape(-1, 1, pred_div_tokens.shape[-1])

        return pred_div_masks, pred_div_ious, pred_div_tokens