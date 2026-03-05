from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class W2VCollator:
    """
    Pads embeddings to max length in batch and creates attention_mask.
    """
    pad_value: float = 0.0

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # embeddings: list of [T_i, D]
        embs = [b["embeddings"] for b in batch]
        lengths = torch.tensor([e.size(0) for e in embs], dtype=torch.long)
        max_len = int(lengths.max().item())
        dim = embs[0].size(1)

        padded = torch.full((len(batch), max_len, dim), self.pad_value, dtype=torch.float32)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, e in enumerate(embs):
            t = e.size(0)
            padded[i, :t, :] = e
            mask[i, :t] = 1

        out: Dict[str, torch.Tensor] = {
            "embeddings": padded,  # [B, T, D]
            "attention_mask": mask,  # [B, T]
            "lengths": lengths  # [B]
        }

        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)

        return out
