# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import hydra
from cell_tracker import SAM2AutomaticCellTracker
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from inference_utils import (
    get_device,
    get_result_path,
    get_tif_directories,
    get_video_path,
)
from omegaconf import OmegaConf

# Local imports
from sam2.build_sam import build_sam2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cell tracking with SAM2")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video file or image sequence directory",
    )
    parser.add_argument(
        "--res_path", type=str, default=None, help="Path to save tracking results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SAM2-tracking-LoRA-heatmap",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--box_nms_thresh",
        type=float,
        default=0.5,
        help="Non-maximum suppression threshold for bounding boxes",
    )
    parser.add_argument(
        "--pred_iou_thresh",
        type=float,
        default=0.7,
        help="IoU threshold for predictions",
    )
    parser.add_argument(
        "--div_obj_score_thresh",
        type=float,
        default=0.5,
        help="Division object score threshold",
    )
    parser.add_argument(
        "--obj_score_thresh",
        type=float,
        default=0.5,
        help="Object score threshold",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Whether to perform segmentation (default: False)",
    )
    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="Disable heatmap (heatmap is used by default)",
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=None, help="Checkpoint number to use"
    )
    parser.add_argument(
        "--resize_threshold",
        type=int,
        default=None,
        help="Enable tiled inference when images exceed this size",
    )
    parser.add_argument(
        "--crop_overlap",
        type=int,
        default=128,
        help="Overlap in pixels between adjacent crops",
    )
    parser.add_argument(
        "--save_crop_movies",
        action="store_true",
        help="Save individual movies for each crop (for debugging)",
    )
    parser.add_argument(
        "--postprocess_divisions",
        action="store_true",
        help="Enable post-processing to detect divisions across crops",
    )
    parser.add_argument(
        "--aux_matching",
        type=str,
        default="off",
        choices=["off", "new_cells_only", "segment_then_aux_track"],
        help="Temporal auxiliary matching: off, new_cells_only (relink new cells via aux head), "
        "segment_then_aux_track (segment each frame + aux track).",
    )
    return parser.parse_args()


def setup_hydra(model_name: str):
    """Setup Hydra configuration.

    Args:
        model_name: Name of the model to use

    """
    config_path = f"../sam2_logs/{model_name}"

    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Global initialization
    hydra.initialize(version_base=None, config_path=config_path)
    return f"../sam2_logs/{model_name}", "config_resolved"


def process_directory(
    tracker: SAM2AutomaticCellTracker,
    dir_path: Path,
    base_dir: Path,
    model_name: str,
    input_path: Path,
    res_path: Path,
    total_dirs: int,
    idx: int,
):
    """Process a single directory with the cell tracker.

    Args:
        tracker: The cell tracker instance
        dir_path: Directory to process
        base_dir: Base directory for results
        model_name: Name of the model
        input_path: Input path being processed
        res_path: Path to save results
        idx: Index of current directory
        total_dirs: Total number of directories

    """
    print(f"Processing directory {idx + 1}/{total_dirs}: {dir_path}")

    # Generate result path
    result_path = get_result_path(
        base_dir=base_dir,
        model_name=model_name,
        input_path=input_path,
        dir_name=dir_path.stem,
        res_path=res_path,
    )
    result_path.mkdir(parents=True, exist_ok=True)

    # Track cells in the video or image sequence
    tracker.predict(
        dir_path,
        result_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        max_frame_num_to_track=None,
    )
    print(f"Finished processing: {dir_path}")


def main():
    """Main function for cell tracking."""
    # Parse arguments and setup paths
    args = parse_args()
    model_name = args.model_name
    config_path, config_name = setup_hydra(model_name)

    # Setup model and device
    device = get_device()
    cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    if args.resize_threshold is None:
        args.resize_threshold = getattr(cfg.scratch, "resize_threshold", None)
    if args.checkpoint_num is None:
        sam2_checkpoint = f"sam2_logs/{model_name}/checkpoints/checkpoint.pt"
    else:
        sam2_checkpoint = (
            f"sam2_logs/{model_name}/checkpoints/checkpoint_{args.checkpoint_num}.pt"
        )
    sam2_model = build_sam2(config_name, sam2_checkpoint, device=device)

    # Create the cell tracker
    tracker = SAM2AutomaticCellTracker(
        sam2_model,
        pred_iou_thresh=args.pred_iou_thresh,
        obj_score_thresh=args.obj_score_thresh,
        div_obj_score_thresh=args.div_obj_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        segment=args.segment,
        use_heatmap=not args.no_heatmap,
        resize_threshold=args.resize_threshold,
        crop_overlap=args.crop_overlap,
        save_crop_movies=args.save_crop_movies,
        postprocess_divisions=args.postprocess_divisions,
        aux_matching=args.aux_matching,
    )

    # Get input path
    video_path = Path(args.video_path) if args.video_path else get_video_path()

    try:
        # Get directories containing .tif files
        directories = get_tif_directories(video_path)

        # Process each directory
        base_dir = Path(__file__).parents[1]
        for idx, dir_path in enumerate(directories):
            process_directory(
                tracker=tracker,
                dir_path=dir_path,
                base_dir=base_dir,
                model_name=model_name,
                input_path=video_path,
                res_path=args.res_path,
                total_dirs=len(directories),
                idx=idx,
            )

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    main()
