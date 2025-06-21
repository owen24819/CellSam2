# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY


def dice_loss(inputs, targets, num_objects):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.flatten(1).mean(-1) / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    return loss / num_objects

def get_border_from_binary_mask(gt_mask: torch.Tensor, size) -> torch.Tensor:
    """
    gt_mask: (H, W) binary tensor
    Returns: (H, W) binary tensor where 1 = border
    """
    gt_mask_resized = F.interpolate(gt_mask.unsqueeze(0)*1.0, size=size, mode='bilinear', align_corners=False).squeeze(0)
    kernel = torch.ones((1, 1, 3, 3), device=gt_mask.device)
    gt_mask_resized = gt_mask_resized.unsqueeze(0).float()  # (1,1,H,W)
    dilated = F.conv2d(gt_mask_resized, kernel, padding=1) > 0  # dilates in-place
    dilated = dilated.float()
    border = dilated - gt_mask_resized  # subtract original to get rim
    return border.squeeze(0).squeeze(0)

def compute_weighted_heatmap_loss(target_masks, heatmap_predictions, target_heatmaps, border_weight=10.0, high_conf_fg_weight=50.0, fg_threshold=0.8):
    """
    Compute weighted binary cross entropy loss for heatmap predictions.
    
    Args:
        target_masks: Ground truth masks tensor
        heatmap_predictions: Predicted heatmap tensor
        target_heatmaps: Target heatmap tensor
    
    Returns:
        Weighted BCE loss for heatmap predictions
    """
    # Get combined target mask and compute border regions
    target_mask = target_masks.max(0).values
    border = get_border_from_binary_mask(target_mask, heatmap_predictions.shape[-2:])
    
    # Initialize weights and apply higher weights to border and high-confidence regions
    weight = torch.ones_like(heatmap_predictions)
    weight = torch.where(border > 0, border_weight, weight)  # Border weight
    weight = torch.where(target_heatmaps > fg_threshold, high_conf_fg_weight, weight)  # High-confidence weight
    
    return F.binary_cross_entropy_with_logits(
        heatmap_predictions, 
        target_heatmaps, 
        weight=weight
    )

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor, target_divide_batch: torch.Tensor, target_heatmaps_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)

        losses = defaultdict(int)
        for outs, targets, target_divide, target_heatmaps in zip(outs_batch, targets_batch, target_divide_batch, target_heatmaps_batch):
            cur_losses = self._forward(outs, targets, target_divide, target_heatmaps)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, target_divide: torch.Tensor, target_heatmaps: torch.Tensor):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_masks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        div_score_logits_list = outputs["multistep_div_score_logits"]
        is_point_used_list = outputs["multistep_is_point_used"]

        pre_div_target_obj_list = outputs["pre_div_target_obj"]
        post_div_target_obj_list = outputs["post_div_target_obj"]

        heatmap_predictions = outputs["heatmap_predictions"]

        loss_heatmap = compute_weighted_heatmap_loss(
            target_masks,
            heatmap_predictions,
            target_heatmaps
        )

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)
        assert len(div_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_div": 0, "loss_class": 0, "loss_heatmap": loss_heatmap}
        for src_masks, ious, object_score_logits, div_score_logits, is_point_used, pre_div_target_obj, post_div_target_obj in zip(src_masks_list, ious_list, object_score_logits_list, div_score_logits_list, is_point_used_list, pre_div_target_obj_list, post_div_target_obj_list):
            target_masks_used = target_masks[is_point_used]
            assert len(target_masks_used) == len(src_masks)

            num_objects = torch.tensor(max(1, src_masks.shape[0]), device=src_masks.device, dtype=torch.float)

            self._update_losses(
                losses, src_masks, target_masks_used, ious, num_objects, object_score_logits, div_score_logits, pre_div_target_obj, post_div_target_obj, target_divide
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, div_score_logits, pre_div_target_obj, post_div_target_obj, target_divide
    ):
        target_masks = target_masks.expand_as(src_masks)

        # get focal, dice and iou loss on all output masks in a prediction step
        loss_mask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        loss_dice = dice_loss(
            src_masks, target_masks, num_objects
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_mask.dtype, device=loss_mask.device
            )
        else:
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                pre_div_target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_div = sigmoid_focal_loss(
            div_score_logits,
            target_divide[:,None].to(torch.float),
            num_objects,
            alpha=self.focal_alpha_obj_score,
            gamma=self.focal_gamma_obj_score,
        )

        loss_iou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            use_l1_loss=self.iou_use_l1_loss,
        )

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * post_div_target_obj
        loss_dice = loss_dice * post_div_target_obj
        loss_iou = loss_iou * post_div_target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_div"] += loss_div.sum()
        losses["loss_class"] += loss_class.sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
