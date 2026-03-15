# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    CTCSegmentLoader,
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]
    man_track: Optional[str] = None

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class CTCRawDataset(VOSRawDataset):
    def __init__(self,
                 train_dir=None,
                 num_frames=None,
                 file_list_txt=None,
                 excluded_videos_list_txt=None,
                 truncate_video=-1,
                 sample_rate=1,
                 target_size=512,
                 resize_threshold=600,
                 training=True,
                 max_samples_per_epoch_per_dir=None,
                 train_sources=None,
                 split="train"):
        if train_sources is not None:
            self.train_dirs, self.max_samples_per_epoch_per_dir = self._resolve_train_sources(
                train_sources, split
            )
        else:
            assert train_dir is not None, "Provide either train_dir or train_sources"
            self.train_dirs = self._resolve_train_dirs(train_dir)
            self.max_samples_per_epoch_per_dir = max_samples_per_epoch_per_dir
        self.num_frames = num_frames
        self.truncate_video = truncate_video
        self.sample_rate = sample_rate
        self.target_size = target_size
        self.resize_threshold = resize_threshold
        self.training = training

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = None

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        self.video_entries = []
        for source_id, train_dir_path in enumerate(self.train_dirs):
            img_folders = list(Path(train_dir_path).glob("[0-9][0-9]"))
            if subset is None:
                video_names = [img_folder.name for img_folder in img_folders]
            else:
                video_names = subset
            video_names = sorted(
                [video_name for video_name in video_names if video_name not in excluded_files]
            )
            for video_name in video_names:
                self.video_entries.append(
                    {
                        "train_dir": Path(train_dir_path),
                        "video_name": video_name,
                        "video_id": len(self.video_entries),
                        "source_id": source_id,
                    }
                )

        # Pre-scan TRA/SEG per video for frame_index and SEG-only single-frame mode
        for entry in self.video_entries:
            gt_root = entry["train_dir"] / (entry["video_name"] + "_GT")
            tra_path = gt_root / "TRA"
            seg_path = gt_root / "SEG"
            has_tra = (tra_path / "man_track.txt").exists()
            has_seg = seg_path.exists()
            if has_tra:
                entry["effective_num_frames"] = self.num_frames
            elif has_seg:
                entry["effective_num_frames"] = 1  # SEG-only: single frame
            else:
                entry["effective_num_frames"] = 0  # Skip: neither TRA nor SEG

        # Build index of (entry_idx, start_idx) pairs
        self.frame_index = []
        for entry_idx, entry in enumerate(self.video_entries):
            all_frames = self.get_all_frames(entry_idx)
            eff_nf = entry["effective_num_frames"]
            if eff_nf == 0:
                continue
            max_start_idx = len(all_frames) - eff_nf + 1 if eff_nf > 1 else len(all_frames)
            for i in range(max_start_idx):
                self.frame_index.append((entry_idx, i))

        self._epoch_index = None
        # Only apply caps if at least one source has a cap; [None, None] means no capping
        caps = self.max_samples_per_epoch_per_dir
        if caps is not None and len(caps) == len(self.train_dirs) and any(c is not None for c in caps):
            self._apply_caps(0)  # train and val both get initial cap at init; val stays fixed, train resamples each set_epoch

    def _apply_caps(self, epoch: int) -> None:
        """Build _epoch_index by capping samples per source (train_dir).
        For training=True uses epoch as seed (new sample each epoch). For val (training=False) uses fixed seed 0 so val set is constant.
        """
        caps = self.max_samples_per_epoch_per_dir
        n_dirs = len(self.train_dirs)
        if caps is None or len(caps) != n_dirs:
            self._epoch_index = None
            return
        by_source = [[] for _ in range(n_dirs)]
        for idx, (entry_idx, start_idx) in enumerate(self.frame_index):
            sid = self.video_entries[entry_idx]["source_id"]
            by_source[sid].append(idx)
        # Val: fixed seed so the same indices every time; train: epoch seed so different sample each epoch
        seed = 0 if not self.training else epoch
        rng = np.random.default_rng(seed)
        self._epoch_index = []
        capped_per = []
        for sid in range(n_dirs):
            indices = np.array(by_source[sid], dtype=np.int64)
            cap = caps[sid]
            if cap is None or cap >= len(indices):
                chosen = indices
            else:
                chosen = rng.choice(indices, size=cap, replace=False)
            capped_per.append(len(chosen))
            for i in np.sort(chosen):
                self._epoch_index.append(self.frame_index[i])
        total = sum(len(x) for x in by_source)
        capped = len(self._epoch_index)
        if capped != total:
            print(f"CTCRawDataset: {'val' if not self.training else f'epoch {epoch}'} capped {total} -> {capped} samples")
        # One-time per-dataset summary before epoch 0 (train and val)
        if epoch == 0 and n_dirs > 1:
            # Path is .../data/<name>/train/CTC or .../data/<name>/val/CTC -> use parts[-3] for <name>
            names = [d.parts[-3] if len(d.parts) >= 3 else (d.parts[-1] if d.parts else str(d)) for d in self.train_dirs]
            for i in range(n_dirs):
                t, c = len(by_source[i]), capped_per[i]
                print(f"  {names[i]}: total={t}" + (f", capped={c}" if c < t else ""))
            print(f"  Total: {total} -> {capped} samples per epoch")

    def set_epoch(self, epoch: int) -> None:
        """Rebuild per-epoch index when using caps. No-op for val so val set stays constant."""
        caps = self.max_samples_per_epoch_per_dir
        if caps is not None and len(caps) == len(self.train_dirs) and any(c is not None for c in caps) and self.training:
            self._apply_caps(epoch)

    def __len__(self):
        index = self._epoch_index if self._epoch_index is not None else self.frame_index
        return len(index)

    def get_all_frames(self, entry_idx):
        """Get a sampled subset of frames from a video.
        Args:
            entry_idx: Index of the video entry
            start_idx: Starting frame index in the sampled sequence
        """
        entry = self.video_entries[entry_idx]
        all_frames = sorted((entry["train_dir"] / entry["video_name"]).glob("*.tif"))
        # Apply sampling first since it reduces video size
        all_frames = all_frames[::self.sample_rate]
        # Then truncate if needed
        if self.truncate_video > 0:
            all_frames = all_frames[:self.truncate_video]            
                        
        return all_frames

    def get_video(self, idx):
        """Get a video starting from the specified frame index"""
        index = self._epoch_index if self._epoch_index is not None else self.frame_index
        entry_idx, start_idx = index[idx]
        entry = self.video_entries[entry_idx]
        train_dir = entry["train_dir"]
        video_name = entry["video_name"]
        
        # Get just the frames we need (use effective_num_frames for SEG-only single-frame)
        all_frames = self.get_all_frames(entry_idx)
        eff_nf = entry["effective_num_frames"]
        selected_frames = all_frames[start_idx : start_idx + eff_nf]
        
        # Create frames list
        frames = []
        for fpath in selected_frames:
            fid = int(re.findall(r'\d+', fpath.stem)[0])
            frames.append(VOSFrame(fid, image_path=fpath))
            
        gt_root = train_dir / (video_name + "_GT")
        tra_path = gt_root / "TRA"
        seg_path = gt_root / "SEG"

        # Load man_track if TRA exists; otherwise SEG-only (no tracking)
        if (tra_path / "man_track.txt").exists():
            man_track = np.loadtxt(tra_path / "man_track.txt", dtype=np.int16)
            # Step 1: Remove parent IDs that appear only once and are positive
            parent_ids, counts = np.unique(man_track[:, -1], return_counts=True)
            single_use_parents = parent_ids[(counts == 1) & (parent_ids > 0)]

            if len(single_use_parents) > 0:
                mask = np.isin(man_track[:, -1], single_use_parents)
                man_track[mask, -1] = 0

            # Step 2: Remove parent IDs whose exit frame is not exactly one before their daughter’s entry
            valid_parents = parent_ids[(counts == 2) & (parent_ids > 0)]

            for parent_id in valid_parents:
                parent_row = man_track[man_track[:, 0] == parent_id]
                daughter_rows = man_track[man_track[:, -1] == parent_id]

                if len(parent_row) == 0 or len(daughter_rows) == 0:
                    continue

                parent_exit_frame = parent_row[0, 2]  # column 2 = end frame
                dau_entry_frame = daughter_rows[:, 1].min()  # column 1 = start frame

                if dau_entry_frame != parent_exit_frame + 1:
                    man_track[man_track[:, -1] == parent_id, -1] = 0

            video_mask_root = tra_path
            mask_prefix = "man_track"
        elif seg_path.exists():
            man_track = None
            video_mask_root = seg_path
            mask_prefix = "man_seg"
        else:
            raise FileNotFoundError(
                f"Neither TRA nor SEG found for {video_name} in {gt_root}"
            )

        video = VOSVideo(video_name, entry["video_id"], frames, man_track)

        first_frame_num = re.findall(r'\d+', selected_frames[0].stem)[0]
        first_mask_path = video_mask_root / (mask_prefix + first_frame_num + ".tif")
        segment_loader = CTCSegmentLoader(
            video_mask_root, 
            first_mask_path, 
            target_size=self.target_size, 
            resize_threshold=self.resize_threshold, 
            training=self.training
            )

        return video, segment_loader

    def _resolve_train_dirs(self, train_dir):
        """Resolve train_dir to a list of paths. Accepts a single path or a list."""
        if isinstance(train_dir, (ListConfig, list, tuple)):
            return [Path(p) for p in train_dir]
        train_dir_path = Path(train_dir)
        if not train_dir_path.exists():
            raise FileNotFoundError(f"train_dir not found: {train_dir_path}")
        return [train_dir_path]

    def _resolve_train_sources(self, train_sources: List[Dict[str, Any]], split: str) -> tuple:
        """Build (train_dirs, max_samples_per_epoch_per_dir) from train_sources.
        Each source: data_dir; cap_train / max_per_epoch for train split; cap_val for val split.
        """
        dirs = [Path(s["data_dir"]) / split / "CTC" for s in train_sources]
        if split == "train":
            caps = [s.get("max_per_epoch_train") or s.get("max_per_epoch") for s in train_sources]
        else:
            caps = [s.get("max_per_epoch_val") for s in train_sources]
        return dirs, caps

class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
