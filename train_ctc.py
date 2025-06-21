import os

import hydra
import torch
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    config_name = "sam2.1_ctc_finetune.yaml"
else:
    config_name = "sam2.1_ctc_finetune_cpu.yaml"
    
config_path = "sam2/configs/sam2.1_training"
model_name = "sam2.1_ctc_segmentationv2"

register_omegaconf_resolvers()

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig) -> None:
    # Use the global model_name variable instead
    global model_name

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", model_name
        )
    else:
        model_name = cfg.launcher.experiment_log_dir.split("/")[-1]

    # Initialize wandb if available and enabled in config
    use_wandb = cfg.scratch.get('use_wandb', False)  # Default to False if not specified
    wandb_config = cfg.get('wandb', {})
    
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WandB logging requested but wandb package is not installed. "
                  "Install with 'pip install wandb' to enable logging.")
        else:
            wandb_run_id = None
            run_id_path = os.path.join(cfg.launcher.experiment_log_dir, "wandb_run_id.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, "r") as f:
                    wandb_run_id = f.read().strip()

            wandb_run = wandb.init(
                project=wandb_config.get('project', 'CellSAM2'),
                name=model_name,
                group=wandb_config.get('group', None),
                config=OmegaConf.to_container(cfg, resolve=True),
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
            )
            
            # Save run ID for future resuming
            makedir(os.path.dirname(run_id_path))
            with open(run_id_path, "w") as f:
                f.write(wandb_run.id)

            # 🛠️ Define how different metrics are tracked
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("val/*", step_metric="val_step")

    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    # add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    local_rank = 0
    world_size = 1
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    trainer = instantiate(cfg.trainer, _recursive_=False)
    
    # Add wandb callback if available and enabled
    if WANDB_AVAILABLE and use_wandb:
        trainer.wandb = wandb
    
    trainer.run()

    # Close wandb run if active
    if WANDB_AVAILABLE and use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()