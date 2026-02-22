"""Lightning module for Whisper-based ASR fine-tuning."""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.text import WordErrorRate

from asr_finetuning.training.config import TrainingConfig
from asr_finetuning.training.utils import decode_batch


def _cosine_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class WhisperModule(pl.LightningModule):
    """LightningModule for Whisper ASR fine-tuning.

    Wraps a pre-constructed model for training with PyTorch Lightning.
    Model construction is handled by build_model() in the model factory.

    Args:
        model: The Whisper model (may have LoRA adapters).
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
        # use_cache=False prevents the decoder from caching KV states, which would
        # otherwise cause issues during training.
        return self.model(input_features=input_features, labels=labels, use_cache=False)

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        if self.config.scheduler == "none":
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        min_lr_ratio = self.config.min_lr / self.config.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _cosine_with_warmup(
                step, self.config.warmup_steps, total_steps, min_lr_ratio
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
