
import os
import yaml
import torch
import torch.nn as nn
from transformers import AutoModel


class LlmCulturalModel(nn.Module):

    def __init__(self,
                 model_name: str,
                 num_labels: int):

        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # ----------------------------------------
        # Case 1: Local checkpoint
        # ----------------------------------------
        if os.path.isdir(model_name) and os.path.exists(
            os.path.join(model_name, "model.pt")
        ):
            self._load_from_local(model_name)

        # ----------------------------------------
        # Case 2: HF model name
        # ----------------------------------------
        else:
            self._load_from_hf(model_name)

        self.loss_fn = nn.CrossEntropyLoss()


    def _load_from_hf(self, model_name):
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)


    def _load_from_local(self, folder):

        with open(os.path.join(folder, "model_config.yaml")) as f:
            cfg = yaml.safe_load(f)

        backbone_name = cfg["model_name"]
        self.num_labels = cfg["num_labels"]

        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Load weights
        state_dict = torch.load(
            os.path.join(folder, "model.pt"),
            map_location="cpu",
        )
        self.load_state_dict(state_dict)


    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}


    def save_model(self,
                   folder_path):

        os.makedirs(folder_path, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), os.path.join(folder_path, "model.pt"))

        # Save model metadata
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
        }

        with open(os.path.join(folder_path, "model_config.yaml"), "w") as f:
            yaml.dump(config, f)