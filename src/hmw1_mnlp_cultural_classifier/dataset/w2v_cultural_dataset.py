

from __future__ import annotations

from typing import Dict

import torch

from hmw1_mnlp_cultural_classifier.dataset.cultural_dataset import CulturalDataset
from hmw1_mnlp_cultural_classifier.dataset.text_builder import CulturalTextBuilder
from hmw1_mnlp_cultural_classifier.embedder.w2v_embedder import W2VEmbedder
from hmw1_mnlp_cultural_classifier.labels_schema.cultural_labels import CulturalLabels
from hmw1_mnlp_cultural_classifier.tokenizer.simple_tokenizer import SimpleTokenizer


class W2VCulturalDataset(CulturalDataset):
    """
    Produces variable-length token-embedding sequences.
    Collation/padding is done in a separate collator.
    """
    def __init__(self,
                 source_dataset_name: str,
                 split: str,
                 hf_key: str,
                 with_labels: bool = True,
                 ):

        super().__init__(source_dataset_name=source_dataset_name,
                         split=split,
                         hf_key=hf_key)

        self.embedder = W2VEmbedder.get_instance("large")
        self.text_builder = CulturalTextBuilder()
        self.tokenizer = SimpleTokenizer()
        self.with_labels = with_labels
        self.labels_schema = CulturalLabels()


    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.source_dataset[idx]
        text = self.text_builder.build(row)
        tokens = self.tokenizer.tokenize(text)

        emb_np = self.embedder.embed_tokens(tokens)  # [T, D]
        embeddings = torch.from_numpy(emb_np)        # FloatTensor [T, D]

        out: Dict[str, torch.Tensor] = {
            "embeddings": embeddings
        }

        if self.with_labels:
            label_name = row["label"]  # per your schema
            out["labels"] = torch.tensor(self.labels_schema.name_to_id[label_name], dtype=torch.long)

        return out
