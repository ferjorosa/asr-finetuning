"""Data collator for speech sequence-to-sequence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator for speech seq2seq models that pads input features and labels.

    Args:
        processor: WhisperProcessor containing feature_extractor and tokenizer.
    """

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Pad input features (audio)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels (text)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding in labels with -100 to ignore during loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # If batch begins with BOS token, remove it from labels
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
