import numpy as np
import torch
from torchvision.ops import batched_nms

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    coco_encode_rle,
    mask_to_rle_pytorch,
    rle_to_mask,
)


class SAM2AutomaticCellSegmenter:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: int = 32,
        points_per_batch: int = 32,
        obj_score_thresh: float = 0.5,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0,  # 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 10,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = False,
        **kwargs,
    ) -> None:
        """Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

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
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.

        """
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.model = model
        self.device = model.device

        if self.device.type == "cpu":
            min_mask_region_area = 0

        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output

    def predict(self, image):
        self.predictor.set_image(image)

        input_points, input_labels = self.generate_proportional_point_grid()

        mask_data = self.predict_in_batches(input_points, input_labels)

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "obj_score": mask_data["obj_scores"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns

    def predict_in_batches(self, input_points, input_labels):
        """Predicts masks in batches to manage memory usage efficiently.

        Args:
            input_points: Point coordinates to predict masks for
            input_labels: Labels corresponding to the points

        Returns:
            masks: Concatenated mask predictions
            scores: Prediction scores for each mask
            logits: Raw logits for each mask

        """
        # Generate masks for this crop in batches
        data = MaskData()

        for batched_points, batched_labels in batch_iterator(
            self.points_per_batch, input_points, input_labels
        ):
            batched_data = self._process_batch(
                points=batched_points,
                labels=batched_labels,
            )

            data.cat(batched_data)
            del batched_data

        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        data.to_numpy()

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        normalize=False,
    ) -> MaskData:
        masks, iou_preds, low_res_masks, obj_scores = self.predictor._predict(
            points,
            labels,
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
            obj_scores=obj_scores.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # One step refinement using previous mask predictions
            in_points = data["points"]

            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious, obj_scores = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)
            data["obj_scores"] = obj_scores.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Compress to RLE
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        new_masks = []
        new_iou_preds = []
        new_obj_scores = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            best_masks, best_iou_preds, _, obj_scores = self.predictor._predict(
                cur_points,
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
            new_obj_scores.append(obj_scores)

        masks = torch.cat(new_masks, dim=0)
        iou_preds = torch.cat(new_iou_preds, dim=0)
        obj_scores = torch.cat(new_obj_scores, dim=0)

        return masks, iou_preds, obj_scores

    def generate_proportional_point_grid(self):
        """Generate a grid of (x, y) points with density proportional to the image size.

        Args:
            image_shape (tuple): (H, W)
            base_density (int): Number of total points if image were 512x512

        Returns:
            points: torch.Tensor of shape [B, N, 2] where each row is (x, y)
            labels: torch.Tensor of shape [B, N]

        """
        H, W = self.predictor.resized_image_size
        scale_factor_y = H / self.model.image_size  # preserve relative density
        scale_factor_x = W / self.model.image_size  # preserve relative density

        # Estimate number of points in each dimension
        points_y = int(self.points_per_side * scale_factor_y)
        points_x = int(self.points_per_side * scale_factor_x)

        # Avoid zero division or very few points
        points_y = max(1, points_y)
        points_x = max(1, points_x)

        offset_y = (self.model.image_size - H) // 2
        offset_x = (self.model.image_size - W) // 2

        # Generate evenly spaced coordinates
        ys = np.linspace(0, H - 1, points_y, dtype=int) + offset_y
        xs = np.linspace(0, W - 1, points_x, dtype=int) + offset_x
        points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

        # Convert points to tensor and add batch dimension
        points = torch.tensor(points, device=self.device, dtype=torch.float32)  # [N, 2]
        points = points.unsqueeze(1)  # [N, 1, 2]

        # Create corresponding labels tensor
        labels = torch.ones(
            len(points), 1, device=self.device, dtype=torch.int
        )  # [N, 1]

        return points, labels
