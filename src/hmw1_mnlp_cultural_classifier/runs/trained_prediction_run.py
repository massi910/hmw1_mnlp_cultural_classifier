import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hmw1_mnlp_cultural_classifier.dataset.cultural_dataset import CulturalDataset
from hmw1_mnlp_cultural_classifier.prediction.cultural_predictor import CulturalPredictor
from hmw1_mnlp_cultural_classifier.utils.device import resolve_device

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="../../../configs", config_name="llm_exp1_cpu")
def start(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.device_name)

    model = instantiate(cfg.model).to(device)

    predictor: CulturalPredictor = CulturalPredictor(
        model=model,
        device_name=cfg.device_name,
        max_seq_len=cfg.max_seq_len,
    )

    validation: CulturalDataset = instantiate(cfg.validation_dataset)

    print(f"Validation dataset size: {len(validation.source_dataset)}")

    for _ in range(3):
        item = validation.source_dataset[_]
        data = validation[_]
        print(f"Source Item {_}: {item}")
        prediction = predictor.predict(
            data=data,
        )

        print(f"Prediction {_}: {prediction}")
        print(f"Ground Truth {_}: {item['label']}")


if __name__ == '__main__':
    start()
