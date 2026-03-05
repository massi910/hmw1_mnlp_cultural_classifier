import os

from HMW1_2025.dataset.cultural_dataset import CulturalDataset
from HMW1_2025.dataset.llm_cultural_dataset import LlmCulturalDataset
from HMW1_2025.model import CulturalModel
from HMW1_2025.model.llm_cultural_model import LlmCulturalModel
from HMW1_2025.prediction.cultural_predictor import CulturalPredictor
from HMW1_2025.tokenizer.base_tokenizer import BaseTokenizer
from HMW1_2025.tokenizer.distil_bert_tokenizer import DistilBertTokenizer

os.environ["HYDRA_FULL_ERROR"] = "1"


def get_model_from_file() -> CulturalModel:
    model_path = "/HMW1_2025/runs/out/train/v1.0.0/pytorch_model.bin"
    model: CulturalModel = LlmCulturalModel.load_trained_model_from_file(
        model_file_path=model_path,
    )

    return model


def get_model_from_url() -> CulturalModel:
    model_url = "https://raw.githubusercontent.com/massi910/models/master/hmw1_nlp_2025/llm_v1.0.0/pytorch_model.bin"
    model: CulturalModel = LlmCulturalModel.load_trained_model_from_url(
        model_url=model_url,
    )

    return model


def start():


    pretrained_name = "distilbert-base-uncased"
    max_seq_len = 64
    device: str = "auto"
    dataset_name = "sapienzanlp/nlp2025_hw1_cultural_dataset"
    split = "validation"

    model: CulturalModel = get_model_from_url()

    tokenizer: BaseTokenizer = DistilBertTokenizer(
        pretrained_name=pretrained_name,
        max_seq_len=max_seq_len
    )

    validation: CulturalDataset = LlmCulturalDataset(
        source_dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer
    )

    predictor: CulturalPredictor = CulturalPredictor(
        model=model,
        device_name=device,
        max_seq_len=max_seq_len,
    )

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
