# Third-party imports
import cv2
import hydra
from hydra.core.global_hydra import GlobalHydra
import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Local imports
from sam2.build_sam import build_sam2
from cell_tracker import SAM2AutomaticCellTracker
from inference_utils import get_device

sys.path.append(str(Path(__file__).parent.parent))

model_name = "SAM2-tracking-LoRA"  
sam2_checkpoint = f"sam2_logs/{model_name}/checkpoints/checkpoint.pt"

config_path = f"../sam2_logs/{model_name}"
config_name = "config_resolved"

# Clear any existing Hydra instance
if GlobalHydra().is_initialized():
    GlobalHydra.instance().clear()

# global initialization
hydra.initialize(version_base=None, config_path=config_path)

device = get_device()

def main(config_name):
    # Build the SAM2 model
    sam2_model = build_sam2(config_name, sam2_checkpoint, device=device)
    
    # Create the cell tracker
    tracker = SAM2AutomaticCellTracker(
        sam2_model,
        points_per_side=16,  # Fewer points for faster processing
        pred_iou_thresh=0.7,
        obj_score_thresh=0,
        div_obj_score_thresh=0,
        stability_score_thresh=0.7,
        box_nms_thresh=0.7,
        min_mask_region_area=30,
        use_m2m=True,  # Use mask-to-mask refinement
        segment=True,
    )

    video_path = 'C:/Users/17742/Documents/DeepLearning/datasets/moma/CTC/test/29'
    res_path = Path(__file__).parents[1] / 'sam2_logs' / model_name / 'results' / 'CTC' / Path(video_path).stem
    res_path.mkdir(parents=True, exist_ok=True)

    if video_path is None:
        # Ask user to select video or image folder
        video_path = get_video_path()

    # Track cells in the video or image sequence
    tracker.predict(
        video_path, 
        res_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    

def get_video_path():
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Ask user if they want to select a video or a folder of images
    selection_window = tk.Toplevel(root)
    selection_window.title("Select Input Type")
    selection_window.geometry("300x150")
    selection_window.resizable(False, False)
    
    selected_option = tk.StringVar(value="video")
    
    tk.Label(selection_window, text="Choose input type:").pack(pady=10)
    tk.Radiobutton(selection_window, text="Video file", variable=selected_option, value="video").pack(anchor=tk.W, padx=20)
    tk.Radiobutton(selection_window, text="Folder of images", variable=selected_option, value="images").pack(anchor=tk.W, padx=20)
    
    path_result = [None]  # Use list to store result from callback
    
    def on_confirm():
        option = selected_option.get()
        if option == "video":
            path_result[0] = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
        else:  # images
            path_result[0] = filedialog.askdirectory(
                title="Select folder containing image sequence"
            )
        selection_window.destroy()
    
    tk.Button(selection_window, text="Confirm", command=on_confirm).pack(pady=20)
    
    # Wait for the window to be closed
    selection_window.wait_window()
    
    if not path_result[0]:
        print("No input selected. Exiting...")
        exit()

    return Path(path_result[0])

if __name__ == "__main__":
    main(config_name) 