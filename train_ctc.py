from hydra.utils import instantiate
import hydra
import os
from pathlib import Path
from omegaconf import OmegaConf
from iopath.common.file_io import g_pathmgr
from training.utils.train_utils import makedir, register_omegaconf_resolvers

config_path = "sam2/configs/sam2.1_training"
config_name = "sam2.1_ctc_finetune.yaml"
model_name = "sam2.1_ctc_segmentationv2"

register_omegaconf_resolvers()

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)

def main(cfg):
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", model_name
        )
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
    trainer.run()

if __name__ == "__main__":
    main()