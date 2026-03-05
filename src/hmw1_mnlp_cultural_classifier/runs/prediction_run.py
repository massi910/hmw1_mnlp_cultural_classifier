import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hmw1_mnlp_cultural_classifier.dataset.cultural_dataset import CulturalDataset
from hmw1_mnlp_cultural_classifier.prediction.cultural_predictor import CulturalPredictor

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="../../../configs", config_name="llm_exp1_cpu")
def start(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    out_file: str = "predictions.jsonl"

    predictor: CulturalPredictor = instantiate(cfg.predictor)
    validation: CulturalDataset = instantiate(cfg.validation_dataset)

    print(f"Validation dataset size: {len(validation.source_dataset)}")

    with open(out_file, "w") as f:
        for _ in range(3):
            item = validation.source_dataset[_]
            data = validation[_]
            print(f"Source Item {_}: {item}")
            prediction = predictor.predict(
                data=data,
            )

            # recreate a new item with the same structure as the source dataset but with the predicted label
            predicted_item = dict(item)
            predicted_item["label"] = prediction[0]

            # write the predicted item to the output file
            f.write(f"{predicted_item}\n")

            print(f"Prediction {_}: {prediction}")
            print(f"Ground Truth {_}: {item['label']}")


if __name__ == '__main__':
    start()
