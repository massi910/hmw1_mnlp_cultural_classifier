from __future__ import annotations

import os
from typing import Dict, Optional
import torch
import torch.nn as nn


class W2VCulturalModel(nn.Module):

    def __init__(
        self,
        num_labels: int,
        embedding_dim: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

        # class weights disabled by default
        self.register_buffer("class_weights", None)

    # ---------------------------------------------------------
    # CLASS WEIGHT HANDLING
    # ---------------------------------------------------------

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """
        Set class weights for cross entropy loss.

        Args:
            weights: Tensor of shape [num_labels]
        """
        if weights.shape[0] != self.num_labels:
            raise ValueError(
                f"class_weights must have size {self.num_labels}, got {weights.shape[0]}"
            )

        weights = weights.float()

        # remove previous buffer if exists
        if "class_weights" in self._buffers:
            del self._buffers["class_weights"]

        # register as buffer so it moves with .to(device)
        self.register_buffer("class_weights", weights)

    def clear_class_weights(self) -> None:
        """Disable class weighting."""
        if "class_weights" in self._buffers:
            del self._buffers["class_weights"]
        self.class_weights = None

    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,          # [B, T, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # masked mean pooling
        if attention_mask is None:
            sent = embeddings.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()      # [B,T,1]
            summed = (embeddings * mask).sum(dim=1)          # [B,D]
            denom = mask.sum(dim=1).clamp(min=1.0)           # [B,1]
            sent = summed / denom

        sent = self.norm(sent)
        x = self.dropout(sent)
        logits = self.classifier(x)

        out = {"logits": logits}

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits,
                labels,
                weight=self.class_weights
            )
            out["loss"] = loss

        return out

    # SAVE / LOAD

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), f"{output_dir}/w2v_model.pt")

    def load_model(self, path: str, map_location="cpu") -> None:

        state = torch.load(path, map_location=map_location)

        if isinstance(state, dict) and "class_weights" in state:
            state.pop("class_weights")

        self.load_state_dict(state, strict=True)
