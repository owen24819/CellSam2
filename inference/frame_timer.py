#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def count_instance_cells(mask: Any, *, threshold: float = 0.5) -> int:
    mask_arr = _coerce_mask(mask)
    if mask_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D mask after squeezing singleton dimensions, got shape {mask_arr.shape}"
        )

    if np.issubdtype(mask_arr.dtype, np.integer):
        labels = np.unique(mask_arr)
        return int(np.count_nonzero(labels > 0))

    binary = mask_arr > threshold
    n_components, _ = cv2.connectedComponents(binary.astype(np.uint8))
    return int(max(0, n_components - 1))


def find_gt_mask_for_frame(gt_seg_dir: Path | None, frame_name: str) -> Path | None:
    if gt_seg_dir is None:
        return None
    digits = re.findall(r"\d+", frame_name)
    if not digits:
        return None
    fid = digits[0]
    for prefix in ("man_seg", "man_track"):
        for suffix in (".tif", ".tiff", ".png"):
            candidate = gt_seg_dir / f"{prefix}{fid}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _coerce_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        mask_arr = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if mask_arr is None:
            raise FileNotFoundError(f"Could not read mask file: {mask}")
    elif hasattr(mask, "detach") and hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(mask)
    return np.squeeze(mask_arr)


class FrameMetricsRecorder:
    def __init__(self, output_csv: str | Path, *, pred_threshold: float = 0.5):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.pred_threshold = pred_threshold

        self._file = open(self.output_csv, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["frame", "seconds", "fps", "gt_cells", "detected_cells"]
        )

    def record_frame(
        self,
        frame_name: str,
        elapsed_seconds: float,
        *,
        gt_mask: Any | None = None,
        pred_mask: Any | None = None,
    ) -> None:
        gt_cells = -1 if gt_mask is None else count_instance_cells(gt_mask)
        detected_cells = (
            -1
            if pred_mask is None
            else count_instance_cells(pred_mask, threshold=self.pred_threshold)
        )

        fps = float("inf") if elapsed_seconds <= 0 else 1.0 / elapsed_seconds
        self._writer.writerow(
            [
                frame_name,
                f"{elapsed_seconds:.6f}",
                f"{fps:.6f}",
                gt_cells,
                detected_cells,
            ]
        )
        self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def record_frame_inference_metrics(
    recorder: FrameMetricsRecorder,
    frame_name: str,
    start_time: float,
    *,
    gt_mask: Any | None = None,
    pred_mask: Any | None = None,
) -> None:
    torch.cuda.synchronize()
    recorder.record_frame(
        frame_name=frame_name,
        elapsed_seconds=time.perf_counter() - start_time,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
    )
