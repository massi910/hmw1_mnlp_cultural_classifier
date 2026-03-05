import os
from typing import Tuple, Dict, Optional, Callable, Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hmw1_mnlp_cultural_classifier.labels_schema.cultural_labels import CulturalLabels
from hmw1_mnlp_cultural_classifier.utils.debugger import Debugger
from hmw1_mnlp_cultural_classifier.utils.device import resolve_device


class W2VCulturalTrainer(Debugger):
    """
    Thin wrapper around Hugging Face Trainer.
    Responsible ONLY for training orchestration.
    """

    def __init__(
            self,
            model,
            train_dataset,
            validation_dataset,
            output_dir: str,
            device_name: str,
            batch_size: int,
            learning_rate: float,
            num_epochs: int,
            collator: Optional[Callable[[Any], Dict[str, Tensor]]] = None,
    ) -> None:

        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.device: torch.device = resolve_device(device_name)

        self.model = model
        self.model.to(self.device)

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.collator = collator

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        self.log(f"Using device: {self.device}")

        os.makedirs(self.output_dir, exist_ok=True)
        self.stats_file: str = f"{self.output_dir}train_stats.tsv"

        # set class weights dynamically from train_dataset
        self._set_class_weights_from_dataset()

    def train(self) -> None:

        # dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collator
        )

        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collator
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

        # dump metrics
        df = pd.DataFrame(all_metrics)
        self.log(f"\n{df}")

        with open(self.stats_file, "w") as f:
            df.to_csv(f, sep="\t", index=False)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        num_classes = self.model.num_labels
        label_counts = torch.zeros(num_classes, dtype=torch.long, device=self.device)

        w0 = None
        if hasattr(self.model, "classifier"):
            w0 = self.model.classifier[-1].weight.detach().clone()

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if "labels" in batch:
                label_counts += torch.bincount(batch["labels"], minlength=num_classes)

            outputs = self.model(**batch)
            loss = outputs["loss"]

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        # --- NEW: log weight movement ---
        if w0 is not None:
            w1 = self.model.classifier[-1].weight.detach()
            delta = (w1 - w0).norm().item()
            wnorm = w1.norm().item()
            self.log(f"[TRAIN] classifier weight norm: {wnorm:.6f}, delta this epoch: {delta:.6f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(
            self,
            loader: DataLoader,
            num_classes: int,
    ) -> Dict[str, float]:
        self.model.eval()

        self.log(f"[DEV] num batches: {len(loader)} ; dev size: {len(self.validation_dataset)}")

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

            # --- DEBUG LOGS (add right here) ---
            if total_samples == 0:  # only log once per epoch (first batch)
                pred_u, pred_c = torch.unique(preds, return_counts=True)
                gold_u, gold_c = torch.unique(labels, return_counts=True)

                self.log(f"[DEV] preds distribution: {dict(zip(pred_u.tolist(), pred_c.tolist()))}")
                self.log(f"[DEV] gold  distribution: {dict(zip(gold_u.tolist(), gold_c.tolist()))}")

                # optional: verify batch shapes
                if "embeddings" in batch:
                    self.log(f"[DEV] embeddings shape: {tuple(batch['embeddings'].shape)}")
                if "attention_mask" in batch:
                    self.log(f"[DEV] attention_mask shape: {tuple(batch['attention_mask'].shape)}")
                self.log(f"[DEV] labels shape: {tuple(labels.shape)}")
            # --- END DEBUG LOGS ---

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            _tp, _fp, _fn = W2VCulturalTrainer.compute_confusion_stats(
                preds, labels, num_classes
            )
            tp += _tp
            fp += _fp
            fn += _fn

        accuracy = total_correct / total_samples

        prec_c = tp / (tp + fp + 1e-8)
        rec_c = tp / (tp + fn + 1e-8)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)

        precision = prec_c.mean().item()
        recall = rec_c.mean().item()
        f1 = f1_c.mean().item()

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

    def _set_class_weights_from_dataset(self) -> None:
        """
        Compute inverse-frequency class weights from the training dataset and set them
        on the model (if the model exposes `set_class_weights`).
        """

        if not hasattr(self.model, "set_class_weights"):
            self.log("[TRAIN] Model has no set_class_weights(); skipping class weighting.")
            return

        num_classes = getattr(self.model, "num_labels", None)
        if num_classes is None:
            self.log("[TRAIN] Model has no num_labels; skipping class weighting.")
            return

        # Try to count labels robustly
        counts = torch.zeros(num_classes, dtype=torch.long)

        labels_schema = CulturalLabels()  # in case labels are strings

        for i in range(len(self.train_dataset)):
            ex = self.train_dataset[i]

            # case 1: typical torch dataset: labels tensor
            if isinstance(ex, dict) and "labels" in ex:
                y = ex["labels"]
                if isinstance(y, torch.Tensor):
                    y = int(y.item())
                else:
                    y = int(y)
                counts[y] += 1

            # case 2: dataset returns raw item with string label (less common here)
            elif isinstance(ex, dict) and "label" in ex:
                yname = ex["label"]
                # map string name -> id
                y = labels_schema.name_to_id[yname]
                counts[y] += 1

            else:
                raise ValueError(
                    "Cannot infer labels from train_dataset[i]. Expected dict with "
                    "'labels' (tensor/int) or 'label' (string)."
                )

        counts_f = counts.float()
        N = counts_f.sum()
        K = float(num_classes)

        # inverse frequency: N / (K * count_c)
        weights = N / (K * torch.clamp(counts_f, min=1.0))

        # set on model (as buffer)
        self.model.set_class_weights(weights.to(self.device))

        self.log(f"[TRAIN] class counts: {counts.tolist()}")
        self.log(f"[TRAIN] class weights: {weights.tolist()}")

