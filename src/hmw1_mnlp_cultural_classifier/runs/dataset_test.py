import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hmw1_mnlp_cultural_classifier.dataset.cultural_dataset import CulturalDataset

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="../../../configs", config_name="llm_exp1")
def start(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train: CulturalDataset = instantiate(cfg.train_dataset)
    validation: CulturalDataset = instantiate(cfg.validation_dataset)

    print(f"Train dataset size: {len(train.source_dataset)}")
    print(f"Validation dataset size: {len(validation.source_dataset)}")

    print(f"Source Item 0: {train.source_dataset[0]}")
    print(f"Item 0: {train[0]}")


if __name__ == '__main__':
    start()
