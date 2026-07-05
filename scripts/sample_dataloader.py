import argparse
import random
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


def _extract_norm(cfg_container):
    def _walk(transforms):
        if isinstance(transforms, dict):
            transforms = [transforms]
        for t in transforms:
            if not isinstance(t, dict):
                continue
            if t.get("_target_", "").endswith("NormalizeAPI"):
                return t.get("mean", [0.485, 0.456, 0.406]), t.get(
                    "std", [0.229, 0.224, 0.225]
                )
            nested = t.get("transforms")
            if nested is not None:
                result = _walk(nested)
                if result is not None:
                    return result
        return None

    mean_std = _walk(cfg_container.get("vos", {}).get("train_transforms", []))
    if mean_std is None:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean_std


def _to_uint8_image(frame_data, mean, std):
    if isinstance(frame_data, Image.Image):
        return np.array(frame_data)
    if isinstance(frame_data, torch.Tensor):
        img = frame_data.detach().cpu()
        if img.dim() == 3:
            mean_t = torch.tensor(mean)[:, None, None]
            std_t = torch.tensor(std)[:, None, None]
            img = img * std_t + mean_t
            img = img.clamp(0, 1)
            img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return img
    raise TypeError("Unsupported frame data type for visualization.")


def _mask_boundary(mask):
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = core & up & down & left & right
    return core & ~interior


def _overlay_masks(image, objects):
    img = image.copy()
    centroids = {}
    parent_to_children = {}

    for obj in objects:
        if not hasattr(obj, "object_id"):
            continue
        if obj.object_id <= 0:
            continue
        mask = obj.segment
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy().astype(bool)
        if mask.sum() == 0:
            continue
        boundary = _mask_boundary(mask)
        color = np.array(
            [
                (obj.object_id * 37) % 255,
                (obj.object_id * 67) % 255,
                (obj.object_id * 97) % 255,
            ],
            dtype=np.uint8,
        )
        img[boundary] = color
        ys, xs = np.where(mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        centroids[obj.object_id] = (cx, cy)

        parent_id = getattr(obj, "parent_id", 0)
        entering = getattr(obj, "entering", False)
        if entering and parent_id > 0:
            parent_to_children.setdefault(parent_id, []).append(obj.object_id)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for obj_id, (cx, cy) in centroids.items():
        draw.text((cx, cy), str(obj_id), fill=(255, 255, 255))

    for child_ids in parent_to_children.values():
        if len(child_ids) != 2:
            continue
        c1 = centroids.get(child_ids[0])
        c2 = centroids.get(child_ids[1])
        if c1 is None or c2 is None:
            continue
        draw.line([c1, c2], fill=(0, 0, 0), width=1)

    return np.array(pil_img)


def main():
    parser = argparse.ArgumentParser(
        description="Sample batches from the training/val dataloader and save augmented frames."
    )
    parser.add_argument("--config_path", type=str, default="../sam2/configs/sam2.1_training")
    parser.add_argument("--config_name", type=str, default="sam2.1_ctc_finetune_cpu.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="temp/debug_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="moma", help="Override scratch.dataset_name")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_path = Path(args.config_path)
    if config_path.is_absolute():
        with initialize_config_dir(version_base=None, config_dir=str(config_path)):
            cfg = compose(config_name=args.config_name)
    else:
        with initialize(version_base=None, config_path=str(config_path)):
            cfg = compose(config_name=args.config_name)

    from training.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    if args.dataset:
        cfg.scratch.dataset_name = args.dataset

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    mean, std = _extract_norm(cfg_container)

    dataset_cfg = cfg.trainer.data[args.split].dataset
    dataset = instantiate(dataset_cfg)

    out_dir = Path(args.out_dir + "_" + args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
    print(f"Saving {len(indices)} samples from {args.split} split to {out_dir.resolve()}")

    for sample_idx, idx in enumerate(indices):
        datapoint = dataset[idx]
        zoom = getattr(datapoint, "clip_zoom_scale", 1.0)
        for t, frame in enumerate(datapoint.frames):
            img = _to_uint8_image(frame.data, mean, std)
            zoom_tag = f"zoom_{zoom:.2f}"
            vis = _overlay_masks(img, frame.objects)
            out_path = out_dir / f"sample_{sample_idx}_idx_{idx}_frame_{t:03d}_{zoom_tag}.png"
            Image.fromarray(vis).save(out_path)
            print(f"  {out_path} (clip_zoom_scale={zoom})")


if __name__ == "__main__":
    main()
