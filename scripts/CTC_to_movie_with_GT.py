import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

dataset = 'DeepCell'  # ['moma','DynamicNuclearNet-tracking-v1_0']
# dataset = 'Fluo-N2DL-HeLa'  # ['moma','DynamicNuclearNet-tracking-v1_0']
basepath = Path('C:/Users/17742/Documents/DeepLearning/data')

splits = ['train', 'val', 'test']
min_height, min_width = 256, 32
fps = 7.0

for split in splits:
    datapath = basepath / dataset / split / 'CTC'
    if not datapath.exists():
        continue
    dataset_ids = [d for d in datapath.iterdir() if re.findall(r'\d\d$', d.name) and int(d.name)]

    all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)])
    colors = {}
    alpha = 0.1

    for index, dataset_id in enumerate(dataset_ids):
        print('='*70)
        print(f'Processing {split} {dataset_id.name} {index+1} of {len(dataset_ids)}...')

        img_fps = sorted([img_fp for img_fp in (datapath / dataset_id.name).iterdir() 
                         if re.findall(r'\d\d\d$', img_fp.stem)])
        man_track = np.loadtxt(datapath / (dataset_id.name + '_GT') / 'TRA' / 'man_track.txt', dtype=np.uint16)
        filepaths = sorted([filepath for filepath in (datapath / (dataset_id.name + '_GT') / 'TRA').iterdir() 
                           if filepath.suffix == '.tif'])
        
        movie = []
        max_pixel = 0
        for idx, img_fp in enumerate(img_fps):
            max_pixel = max(max_pixel, np.max(cv2.imread(str(img_fp), cv2.IMREAD_ANYDEPTH)))
            instance = cv2.imread(str(filepaths[idx]), cv2.IMREAD_ANYDEPTH)
            img = cv2.imread(str(img_fp), cv2.IMREAD_ANYDEPTH)
            img = np.stack((img, img, img), axis=-1)
            img = (img - np.min(img)) / max_pixel
            img = (img * 255).astype(np.uint8)

            cellnbs = np.unique(instance)
            cellnbs = cellnbs[cellnbs != 0]
            daus = []

            for cellnb in cellnbs:
                cellnb_ind = man_track[:,0] == cellnb
                assert cellnb in man_track[:,0]
                if man_track[cellnb_ind,1] == idx and man_track[cellnb_ind,-1] > 0:
                    mother_id = man_track[cellnb_ind,-1][0]
                    dau_ids = man_track[man_track[:,-1] == mother_id, 0]
                    if len(dau_ids) != 2:
                        if cellnb not in colors.keys():
                            colors[cellnb] = all_colors[cellnb]
                    else:
                        dau_1, dau_2 = dau_ids
                        daus.append([dau_1, dau_2])

                        if cellnb == dau_1:
                            if np.where(instance == dau_1)[0].mean() < np.where(instance == dau_2)[0].mean():
                                colors[cellnb] = colors[mother_id]
                            else:
                                colors[cellnb] = all_colors[cellnb]
                        elif cellnb == dau_2:
                            if np.where(instance == dau_2)[0].mean() < np.where(instance == dau_1)[0].mean():
                                colors[cellnb] = colors[mother_id]
                            else:
                                colors[cellnb] = all_colors[cellnb]
                        else:
                            pass
                else: 
                    if cellnb not in colors.keys():
                        colors[cellnb] = all_colors[cellnb]

                color_mask = np.zeros_like(img)
                color_mask[instance == cellnb] = colors[cellnb]
                img[instance == cellnb] = alpha * color_mask[instance==cellnb] + (1-alpha) * img[instance == cellnb]

                # contour with same color as mask for visibility with low alpha
                cell_mask = (instance == cellnb).astype(np.uint8)
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_color = tuple(int(c) for c in colors[cellnb][::-1])  # RGB -> BGR
                img = cv2.drawContours(img, contours, -1, contour_color, thickness=1)

                cell_loc = np.where(instance == cellnb)
                cell_loc = [int(np.median(cell_loc[0])), int(np.median(cell_loc[1]))]

                fontscale = 0.4
                if img.shape[1] < 100:
                    org = (max(cell_loc[1] - (img.shape[1] // 3) * int(np.log10(cellnb)),0), cell_loc[0])
                else:
                    org = (cell_loc[1], cell_loc[0])
                img = cv2.putText(
                    img,
                    text = str(cellnb), 
                    org=org, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontscale,
                    color = (0,0,0) if max_pixel < 256 else (255,255,255),
                    thickness=1,
                    )

            for div in daus:
                w1 = np.where(instance == div[0])
                w2 = np.where(instance == div[1])
                if w1[0].size == 0 or w2[0].size == 0:
                    continue  # cell not in this frame, skip arrow
                cell_1 = [int(np.median(w1[0])), int(np.median(w1[1]))]
                cell_2 = [int(np.median(w2[0])), int(np.median(w2[1]))]

                if cell_1[0] > cell_2[0]:
                    cell_1, cell_2 = cell_2, cell_1

                img = cv2.arrowedLine(img, (cell_1[1], cell_1[0]), (cell_2[1], cell_2[0]), 
                                      color=(0, 0, 0), thickness=1)

            img = cv2.putText(img, text=str(idx), org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                             fontScale=fontscale, color=(255,255,255), thickness=1)
            img = cv2.putText(img, text='GT', org=(0,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                             fontScale=fontscale, color=(255,255,255), thickness=1)

            movie.append(img)

        # Resize all frames: use each movie's dimensions, with minimum floor
        movie = np.stack(movie, axis=0)
        h, w = movie.shape[1], movie.shape[2]
        target_height = max(h, min_height)
        target_width = max(w, min_width)
        resized_movie = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) 
                        for frame in movie]
        movie = np.stack(resized_movie, axis=0)

        # Write to temporary raw file
        filename = dataset_id / 'gt_movie.mp4'
        temp_raw = filename.with_suffix('.raw')
        expected_frame_size = target_width * target_height * 3
        
        with open(temp_raw, 'wb') as f:
            for frame in movie:
                frame_bytes = np.ascontiguousarray(frame.astype(np.uint8)).tobytes()
                f.write(frame_bytes)

        # Encode to MP4 using ffmpeg
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found in PATH")
        
        duration_seconds = len(movie) / fps
        ffmpeg_cmd = [
            ffmpeg_path, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{target_width}x{target_height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-t', f'{duration_seconds:.6f}', '-i', str(temp_raw),
            '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            '-profile:v', 'baseline', '-level', '3.0', '-crf', '23',
            '-preset', 'medium', '-r', str(fps), '-g', str(int(fps * 2)), '-bf', '0',
            '-loglevel', 'error', str(filename)
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Cleanup
        if temp_raw.exists():
            temp_raw.unlink()
        
        file_size = filename.stat().st_size
        print(f"  ✓ Saved: {filename.name} ({file_size} bytes, {len(movie)} frames)")
