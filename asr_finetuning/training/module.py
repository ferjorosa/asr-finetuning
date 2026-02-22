"""Lightning module for Whisper-based ASR fine-tuning."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.text import WordErrorRate

from asr_finetuning.training.config import TrainingConfig
from asr_finetuning.training.utils import decode_batch


class WhisperModule(pl.LightningModule):
    """LightningModule for Whisper ASR fine-tuning.

    Wraps a pre-constructed model (PEFT-wrapped or full) for training with
    PyTorch Lightning. Model construction is handled by build_model() in
    the model factory.

    Args:
        model: The Whisper model (may be PEFT-wrapped).
        processor: WhisperProcessor for decoding and evaluation.
        config: Training configuration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self.val_wer: WordErrorRate = WordErrorRate()

    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> Any:
        return self.model(input_features=input_features, labels=labels)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.forward(
            input_features=batch["input_features"],
            labels=batch["labels"],
        ).loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.forward(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        self.log("val_loss", outputs.loss, prog_bar=True, on_step=False, on_epoch=True)
        preds, refs = decode_batch(outputs.logits, batch["labels"], self.processor)
        self.val_wer.update(preds, refs)  # ty: ignore[invalid-argument-type]

    def on_validation_epoch_end(self) -> None:
        self.log("val_wer", self.val_wer.compute() * 100, prog_bar=True)  # ty: ignore[missing-argument]
        self.val_wer.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
