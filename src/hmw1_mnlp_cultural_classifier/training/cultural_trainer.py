import os
from typing import Tuple, Dict

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hmw1_mnlp_cultural_classifier.dataset.llm_cultural_dataset import LlmCulturalDataset
from hmw1_mnlp_cultural_classifier.model.llm_cultural_model import LlmCulturalModel
from hmw1_mnlp_cultural_classifier.utils.debugger import Debugger
from hmw1_mnlp_cultural_classifier.utils.device import resolve_device


class CulturalTrainer(Debugger):
    """
    Thin wrapper around Hugging Face Trainer.
    Responsible ONLY for training orchestration.
    """


    def __init__(
            self,
            model: LlmCulturalModel,
            train_dataset: LlmCulturalDataset,
            validation_dataset: LlmCulturalDataset,
            output_dir: str,
            device_name: str,
            batch_size: int,
            learning_rate: float,
            num_epochs: int,
    ) -> None:

        super().__init__()
        self.train_dataset: LlmCulturalDataset = train_dataset
        self.validation_dataset: LlmCulturalDataset = validation_dataset
        self.device: torch.device = resolve_device(device_name)
        self.model: LlmCulturalModel = model
        model.to(self.device)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        self.log(f"Using device: {self.device}")

        os.makedirs(self.output_dir, exist_ok=True)
        self.stats_file: str = f"{self.output_dir}train_stats.tsv"

    def train(self) -> None:

        # dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        num_classes = self.model.num_labels

        all_metrics = []
        for epoch in range(1, self.num_epochs + 1):
            self.log(f"Epoch {epoch}/{self.num_epochs}")

            train_loss = self._train_epoch(train_loader)
            val_metrics = self._eval_epoch(val_loader, num_classes)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss
            }
            metrics.update(val_metrics)

            all_metrics.append(metrics)

        self.model.save_model(self.output_dir)
        #self.model.save_pretrained(self.output_dir+"model")

        # dump metrics
        df = pd.DataFrame(all_metrics)
        self.log(f"\n{df}")

        with open(self.stats_file, "w") as f:
            df.to_csv(f, sep="\t", index=False)





    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs["loss"]

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(
            self,
            loader: DataLoader,
            num_classes: int,
    ) -> Dict[str, float]:
        self.model.eval()

        total_correct = 0
        total_samples = 0

        tp = torch.zeros(num_classes, device=self.device)
        fp = torch.zeros(num_classes, device=self.device)
        fn = torch.zeros(num_classes, device=self.device)

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            logits = outputs["logits"]
            labels = batch["labels"]

            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            _tp, _fp, _fn = CulturalTrainer.compute_confusion_stats(
                preds, labels, num_classes
            )
            tp += _tp
            fp += _fp
            fn += _fn

        accuracy = total_correct / total_samples

        precision = (tp / (tp + fp + 1e-8)).mean().item()
        recall = (tp / (tp + fn + 1e-8)).mean().item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def compute_confusion_stats(
            preds: Tensor,
            labels: Tensor,
            num_classes: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns per-class (tp, fp, fn)
        """
        tp = torch.zeros(num_classes, device=preds.device)
        fp = torch.zeros(num_classes, device=preds.device)
        fn = torch.zeros(num_classes, device=preds.device)

        for c in range(num_classes):
            tp[c] = ((preds == c) & (labels == c)).sum()
            fp[c] = ((preds == c) & (labels != c)).sum()
            fn[c] = ((preds != c) & (labels == c)).sum()

        return tp, fp, fn

