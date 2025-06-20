### Model Comparisons
- [ ] Compare LoRA vs full fine-tune on MOMA dataset

# Training Code for CellSAM2

This folder contains the training code for SAM 2, a foundation model for promptable visual segmentation in images and videos. 
The code allows users to train and fine-tune SAM 2 on their own datasets (image, video, or both).

## 🚀 Getting Started

| Task        | Link |
|-------------|------|
| 🔧 Training | [Colab Notebook](https://colab.research.google.com/drive/1I9HCPukpnXtm0HW7isccpsdAnOjkt0mo#scrollTo=BxvFxpDehhR6) |
| 📈 Inference | [Colab Notebook - Coming Soon]() |

#### Requirements:
- Download the moma dataset from [zenodo](https://zenodo.org/records/11237127).

## Data Format

CellSAM2 supports data in the Cell Tracking Challenge (CTC) format. Your data should follow the CTC structure:

```
data/
├── moma/                    # moma dataset
│   ├── train/
│   │   └── CTC/             # CTC format folder
│   │       ├── 01/          # Sequence folder
│   │       │   ├── t000.tif # Time point images
│   │       │   └── ...
│   │       ├── 01_GT/       # Ground truth folder
│   │       │   ├── TRA/     # Tracking annotations (used for training)
│   │       │   │   ├── man_track000.tif
│   │       │   │   ├── ...
│   │       │   │   └── man_track.txt
│   │       │   └── SEG/     # Segmentation masks
│   │       │       ├── man_seg000.tif
│   │       │       └── ...
│   │       └── ...
│   ├── val/
│   │   └── CTC/             # Same structure as train/CTC
│   └── test/
│       └── CTC/             # Same structure as train/CTC
└── DynamicNuclearNet/        # DynamicNuclearNet dataset
    ├── train/
    │   └── CTC/             # Same structure as moma/train/CTC
    ├── val/
    │   └── CTC/             # Same structure as moma/train/CTC
    └── test/
        └── CTC/             # Same structure as moma/train/CTC
```

#### Steps to fine-tune on the moma dataset:
- Install the packages required for training by running `pip install -e ".[dev]"`.
- Set the paths for moma dataset in `configs/sam2.1_training/sam2.1_ctc_finetune.yaml`.
    ```yaml
    dataset:
        # PATHS to Dataset
        data_dir: null # PATH to moma dataset
    ```

# CellSAM2 Training Guide

This guide provides detailed information about training CellSAM2 for cell segmentation and tracking tasks.

## Structure

The training code is organized into the following subfolders:

* `dataset`: This folder contains image and video dataset and dataloader classes as well as their transforms.
* `model`: This folder contains the main model class (`SAM2Train`) for training/fine-tuning. `SAM2Train` inherits from `SAM2Base` model and provides functions to enable training or fine-tuning SAM 2.
* `utils`: This folder contains training utils such as loggers and distributed training utils.
* `loss_fns.py`: This file has the main loss class (`MultiStepMultiMasksAndIous`) used for training.
* `optimizer.py`:  This file contains all optimizer utils that support arbitrary schedulers.
* `trainer.py`: This file contains the `Trainer` class that accepts all the `Hydra` configurable modules (model, optimizer, datasets, etc..) and implements the main train/eval loop.
* `train_ctc.py`: This script is used to launch training jobs.

## Training Commands

### Basic Training

You can train CellSAM2 for either tracking or segmentation tasks. The default mode is tracking.

```bash
# For cell tracking (default)
python train_ctc.py \
  launcher.experiment_log_dir=sam2_logs/CellSam2-tracking \
  scratch.dataset_name={dataset} \
  dataset.data_dir={dataset_path}

# For cell segmentation (single frame)
python train_ctc.py \
  launcher.experiment_log_dir=sam2_logs/CellSam2-tracking \
  scratch.dataset_name={dataset} \
  dataset.data_dir={dataset_path} \
  scratch.num_frames=1
```

## Configuration Options

CellSAM2 provides several configuration options to customize training:

### Required Parameters
- `launcher.experiment_log_dir`: Directory for saving experiment logs and checkpoints

### Basic Parameters
- `scratch.num_epochs`: Number of training epochs (default: 5)
- `scratch.num_frames`: Number of frames to process (set to 1 for segmentation mode, default: 4)
- `scratch.max_num_objects`: Maximum number of cells to track/segment (default: 30)
- `scratch.batch_size`: Batch size for training (default: 1)
- `scratch.resolution`: Input image resolution (default: 512)

### Training Mode and Learning Rates
- `scratch.use_lora`: Use LoRA for efficient fine-tuning (default: True)
  - When LoRA is enabled:
    - `scratch.lora_lr`: Learning rate for LoRA parameters (default: 1.0e-3)
    - Model outside LoRA parameters will be frozen automatically
  - When LoRA is disabled:
    - `scratch.base_lr`: Base learning rate (default: 5.0e-6)
    - `scratch.vision_lr`: Vision encoder learning rate (default: 3.0e-6)
    - `scratch.freeze_encoder`: Whether to freeze the encoder (default: False)

### Dataset Parameters
- `scratch.max_num_bkgd_objects`: Maximum number of background objects (default: 8)
- `dataset.data_dir`: Path to your dataset directory (default: ../data/${scratch.dataset_name})

### Logging
- `scratch.use_wandb`: Enable Weights & Biases logging (default: False)


**Note:** Only the TRA (tracking annotations) folder is used for tracking tasks
CellSAM2 only works for batch size of 1. You will get an error if you increase batch size. Future work will address this.

## Training Tips

1. **Start with LoRA**: Use `scratch.use_lora=True` for faster training and lower memory usage
2. **Adjust batch size**: Increase `scratch.batch_size` if you have sufficient GPU memory
3. **Monitor training**: Enable Weights & Biases logging with `scratch.use_wandb=True` for better tracking
4. **Data augmentation**: The training pipeline includes various augmentations by default
5. **Checkpointing**: Models are automatically saved every epoch during training in the experiment log directory

## Troubleshooting

- **Out of memory**: Reduce `scratch.max_num_objects` then `scratch.num_frames`
- **Poor convergence**: Ensure your images are getting read correctly, it's only been on two datasets so far
