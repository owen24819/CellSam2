import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def get_device():
    """Get the device to use for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_video_path() -> Path:
    """Open a GUI dialog for selecting video file or image sequence directory.
    
    Returns:
        Path: Selected path to video file or image directory
    """
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

def get_result_path(base_dir: Path, model_name: str, input_path: Path, dir_name: str, res_path: Path = None) -> Path:
    """Generate the result path based on input path structure.
    
    Args:
        base_dir: Base directory (usually __file__.parents[1])
        model_name: Name of the model being used
        input_path: Input path being processed
        dir_name: Name of directory being processed
        res_path: Result path to save to

    Returns:
        Path: Result directory path
    """
    if res_path is not None:
        return Path(res_path) / dir_name
    
    # Start with common base path
    result_path = base_dir / 'sam2_logs' / model_name / 'results'
    
    # Add split directory if in train/val/test
    if 'test' in input_path.parts:
        result_path = result_path / 'test'
    elif 'train' in input_path.parts:
        result_path = result_path / 'train'
    elif 'val' in input_path.parts:
        result_path = result_path / 'val'
    
    # Add CTC directory if in CTC dataset
    if 'CTC' in input_path.parts:
        result_path = result_path / 'CTC'
    
    # Add final directory name
    return result_path / dir_name

def has_tif_files(path):
    """Check if the directory contains .tif files."""
    path = Path(path)
    return any(f.suffix.lower() == '.tif' for f in path.glob('*.[tT][iI][fF]'))

def get_tif_directories(base_path):
    """Get all directories containing .tif files."""
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise ValueError(f"{base_path} is not a directory")
    
    # If the base directory has .tif files, return just that
    if has_tif_files(base_path):
        return [base_path]
    
    # Otherwise, look for subdirectories with .tif files
    tif_dirs = []
    for subdir in base_path.iterdir():
        if subdir.is_dir() and has_tif_files(subdir):
            tif_dirs.append(subdir)
    
    if not tif_dirs:
        raise ValueError(f"No directories containing .tif files found in {base_path}")
    
    return sorted(tif_dirs)

def display_masks(image, masks):

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 

def show_anns(anns, borders=True, mask_alpha=0.1):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [mask_alpha]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
            
            # Add IoU prediction text
            # Get centroid of the largest contour to place text
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Add text with IoU value
                plt.text(cx, cy, f'IoU: {ann["predicted_iou"]:.2f}\nObj Score: {ann["obj_score"]:.2f}\nStability: {ann["stability_score"]:.2f}', 
                        color='white', fontsize=8, 
                        bbox=dict(facecolor='black', alpha=0.5))

    ax.imshow(img)