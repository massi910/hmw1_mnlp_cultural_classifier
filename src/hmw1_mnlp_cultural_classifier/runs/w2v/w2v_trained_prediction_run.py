import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from HMW1_2025.dataset.cultural_dataset import CulturalDataset
from HMW1_2025.dataset.w2v_cultural_dataset import W2VCulturalDataset
from HMW1_2025.model.w2v_cultural_model import W2VCulturalModel
from HMW1_2025.prediction.cultural_predictor import CulturalPredictor
from HMW1_2025.utils.device import resolve_device

os.environ["HYDRA_FULL_ERROR"] = "1"


if __name__ == '__main__':

    num_labels: int = 3
    device: str = "auto"
    source_dataset_name: str = "sapienzanlp/nlp2025_hw1_cultural_dataset"
    split: str = "validation"
    with_labels: bool = True
    trained_model_path: str = "/Users/massi910/repo/HMW1_2025/src/HMW1_2025/runs/w2v/out/train/w2v/v1.0.0/w2v_model.pt"

    model: W2VCulturalModel = W2VCulturalModel(num_labels=num_labels)

    print("Loading model from:", trained_model_path)
    model.load_model(trained_model_path)

    predictor: CulturalPredictor = CulturalPredictor(
        model=model,
        device_name=device,
    )

    dataset: CulturalDataset = W2VCulturalDataset(source_dataset_name=source_dataset_name, split=split, hf_key="")

    out_file: str = "predictions.jsonl"

    with open(out_file, "w") as f:
        for _ in range(len(dataset)):
            item = dataset.source_dataset[_]
            data = dataset[_]
            print(f"Source Item {_}: {item}")
            prediction = predictor.predict(
                data=data,
            )

            # recreate a new item with the same structure as the source dataset but with the predicted label
            predicted_item = dict(item)
            predicted_item["label"] = prediction[0]

            # write the predicted item to the output file
            f.write(f"{predicted_item}\n")

