from typing import Dict, Any

import torch
from torch import Tensor

from hmw1_mnlp_cultural_classifier.dataset.cultural_dataset import CulturalDataset
from hmw1_mnlp_cultural_classifier.tokenizer.base_tokenizer import BaseTokenizer


class LlmCulturalDataset(CulturalDataset):

    def __init__(self,
                 source_dataset_name: str,
                 split: str,
                 tokenizer: BaseTokenizer,
                 hf_key: str
                 ):
        super().__init__(source_dataset_name=source_dataset_name,
                         split=split,
                         hf_key=hf_key
                         )
        self.tokenizer = tokenizer

        data: Dict[str, list[Any]] = self._preprocess(self.source_dataset)
        texts = data["texts"]
        self.encodings: Dict[str, Tensor] = self.tokenizer.encode(texts)
        self.labels = data["labels"]


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


    @staticmethod
    def build_text(row):
        return (
            f"Name: {row['name']}. "
            f"Description: {row['description']}. "
            f"Type: {row['type']}. "
            f"Category: {row['category']}. "
            f"Subcategory: {row['subcategory']}."
        )


    def _preprocess(self, dataset) -> Dict[str, list]:
        """
        Takes a Hugging Face Dataset object and returns
        a dict of text & labels_schema lists
        """
        texts = []
        labels = []

        for row in dataset:
            texts.append(LlmCulturalDataset.build_text(row))
            labels.append(self.labels_schema.name_to_id[row["label"]])

        return {"texts": texts, "labels": labels}


