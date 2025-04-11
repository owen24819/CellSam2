# Third-party imports
from PIL import Image
import hydra
from hydra.core.global_hydra import GlobalHydra
import tkinter as tk
from tkinter import filedialog

# Local imports
from sam2.build_sam import build_sam2
from cell_segmenter import SAM2AutomaticCellSegmenter
from inference_utils import display_masks, get_device

model_name = "sam2.1_ctc_segmentation"  
sam2_checkpoint = f"sam2_logs/{model_name}/checkpoints/checkpoint.pt"
# config_path = f"sam2_logs/{model_name}"  
# config_name = "config_resolved"

config_path = "../sam2/configs/sam2.1"  
config_name = "sam2.1_hiera_b+"

# Clear any existing Hydra instance
if GlobalHydra().is_initialized():
    GlobalHydra.instance().clear()

# global initialization
hydra.initialize(version_base=None, config_path=config_path)

device = get_device()

def main(config_name):

    sam2_model = build_sam2(config_name, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2AutomaticCellSegmenter(sam2_model)

    image_path = None

    if image_path is None:
        # Ask user to select image from folder
        image_path = get_image_path()


    image = Image.open(image_path).convert("RGB")
    masks = predictor.predict(image)
    display_masks(image, masks)

def get_image_path():
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog for image selection
    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("No image selected. Exiting...")
        exit()

    return image_path

if __name__ == "__main__":
    main(config_name)
