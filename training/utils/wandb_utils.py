"""Optional W&B logging. Enable with scratch.use_wandb=true and pip install wandb."""

import os
from typing import Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[misc, assignment]
    WANDB_AVAILABLE = False


def init_wandb(cfg: Any, model_name: str, log_dir: str) -> None:
    """Initialize a W&B run with resume support and default metrics. Call only when use_wandb is True."""
    from omegaconf import OmegaConf

    from training.utils.train_utils import makedir

    run_id_path = os.path.join(log_dir, "wandb_run_id.txt")
    run_id = None
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
    wandb.init(
        project=cfg.get('wandb', {}).get('project', 'CellSAM2'),
        name=model_name,
        group=cfg.get('wandb', {}).get('group'),
        config=OmegaConf.to_container(cfg, resolve=True),
        id=run_id,
        resume="allow" if run_id else None,
    )
    makedir(log_dir)
    with open(run_id_path, "w") as f:
        f.write(wandb.run.id)
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("val/*", step_metric="val_step")
