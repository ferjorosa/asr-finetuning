"""Training utilities for Whisper-based ASR fine-tuning."""

from typing import Any

import torch


def decode_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    processor: Any,
    ignore_pad: bool = True,
) -> tuple[list[str], list[str]]:
    """Decode a batch of logits and labels to text strings.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Ground truth label token IDs of shape (batch, seq_len).
        processor: WhisperProcessor for decoding token IDs to text.
        ignore_pad: Whether to replace -100 (padding) with pad_token_id before decoding.

    Returns:
        Tuple of (predictions, references) as lists of strings.
    """
    if ignore_pad:
        labels = labels.clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    return pred_str, label_str
