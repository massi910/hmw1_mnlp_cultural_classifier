
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from hmw1_mnlp_cultural_classifier.labels_schema.cultural_labels import CulturalLabels
from hmw1_mnlp_cultural_classifier.utils.device import resolve_device


class CulturalPredictor:
    """
    Device-aware inference wrapper for cultural classification.
    """

    def __init__(
            self,
            model: nn.Module,
            device_name: str,
            max_seq_len: int = 64,
            model_input_keys: Optional[List[str]] = None,
    ) -> None:

        self.device = resolve_device(device_name)
        self.model = model.to(self.device)
        self.model.eval()

        self.labels_schema: CulturalLabels = CulturalLabels()
        self.max_seq_len = max_seq_len
        self.model_input_keys = model_input_keys or ["input_ids", "attention_mask"]

    @torch.no_grad()
    def predict(self, data: Dict[str, Tensor]) -> List[str]:

        # ---- LM path (existing) OR W2V path (new) ----
        if "input_ids" in data:
            input_ids = data["input_ids"]
            attention_mask = data.get("attention_mask", None)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        else:
            # ---- W2V path ----
            embeddings = data["embeddings"]  # [T,D] or [B,T,D]
            attention_mask = data.get("attention_mask", None)

            if embeddings.dim() == 2:  # [T,D] -> [1,T,D]
                embeddings = embeddings.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            embeddings = embeddings.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(embeddings=embeddings, attention_mask=attention_mask)

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return [self.labels_schema.id_to_name[int(p)] for p in preds]

