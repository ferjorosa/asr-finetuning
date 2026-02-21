"""Dataset for ASR fine-tuning."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from asr_finetuning.data.config import DataConfig


class ASRDataset(Dataset[dict[str, Any]]):
    """Wraps a dataset for ASR fine-tuning by extracting audio features and tokenizing text.

    Expects examples with an audio column (containing 'array' and 'sampling_rate')
    and a text column. Column names are read from DataConfig.

    Args:
        hf_dataset: Any indexable dataset with a known length.
        processor: WhisperProcessor for feature extraction and tokenization.
        config: Data configuration (column names, sampling rate).
    """

    def __init__(self, hf_dataset: Any, processor: Any, config: DataConfig) -> None:
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:  # type: ignore[override]
        example = self.hf_dataset[idx]

        audio_array = example[self.config.audio_column]["array"]
        sampling_rate = example[self.config.audio_column]["sampling_rate"]
        features = self.processor.feature_extractor(
            audio_array, sampling_rate=sampling_rate
        )

        tokenized_text = self.processor.tokenizer(example[self.config.text_column])

        return {
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        }
