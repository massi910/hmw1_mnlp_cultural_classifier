import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hmw1_mnlp_cultural_classifier.training.cultural_trainer import CulturalTrainer

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="../../../configs", config_name="llm_exp1_cpu")
def start(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer: CulturalTrainer = instantiate(cfg.trainer)

    trainer.train()


if __name__ == '__main__':
    start()
