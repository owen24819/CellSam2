import os

import hydra
import torch
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers
from training.utils.wandb_utils import WANDB_AVAILABLE, init_wandb, wandb

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

    # Optional: set scratch.use_wandb=true and add wandb.project/group in config to log to W&B
    use_wandb = cfg.scratch.get('use_wandb', False) and WANDB_AVAILABLE
    if cfg.scratch.get('use_wandb', False) and not WANDB_AVAILABLE:
        print("WandB requested but not installed. pip install wandb to enable.")
    if use_wandb:
        init_wandb(cfg, model_name, cfg.launcher.experiment_log_dir)

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
    if use_wandb:
        trainer.wandb = wandb
    trainer.run()
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()