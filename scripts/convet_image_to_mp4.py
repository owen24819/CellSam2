from tkinter import Tk, filedialog

import cv2
from PIL import Image


def select_images():
    Tk().withdraw()  # Hide the root window
    filetypes = [("*.jpg *.jpeg *.png *.bmp", "*.tif")]
    filenames = filedialog.askopenfilenames(title="Select images", filetypes=filetypes)
    return list(filenames)

def make_video(image_paths, output_path="output_video.mp4", fps=10):
    if not image_paths:
        print("No images selected.")
        return

    # Sort images to preserve order
    image_paths.sort()

    # Load first image to get size
    img = Image.open(image_paths[0])
    width, height = img.size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in image_paths:
        img = Image.open(path).resize((width, height))

        if img.mode == "RGB":
            frame = np.array(img)
        elif img.mode == "I;16":
            arr = np.array(img)

            # Get 1st and 99th percentiles for robust scaling
            p1, p99 = np.percentile(arr, [1, 99])
            
            # Handle case where all values are identical (including all zeros)
            if p1 == p99:
                arr_8bit = np.zeros_like(arr, dtype=np.uint8)
            else:
                # Clip values to percentiles and scale to 0-255
                arr_clipped = np.clip(arr, p1, p99)
                arr_8bit = ((arr_clipped - p1) * (255.0 / (p99 - p1))).astype(np.uint8)
            frame = np.stack([arr_8bit]*3, axis=-1)
        else:
            raise ValueError(f"Unexpected image mode: {img.mode}. Please inspect and handle this mode manually.")

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

def select_save_location():
    Tk().withdraw()  # Hide the root window
    filename = filedialog.asksaveasfilename(
        title="Save video as",
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4")]
    )
    return filename


if __name__ == "__main__":
    import numpy as np

    image_paths = select_images()
    if image_paths:
        output_path = select_save_location()
        if output_path:
            make_video(image_paths, output_path)
        else:
            print("Video save cancelled.")

