"""Data configuration for ASR fine-tuning."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Schema and split configuration for ASR datasets.

    Args:
        audio_column: Name of the column containing audio data.
        text_column: Name of the column containing text transcriptions.
        sampling_rate: Target sampling rate for audio resampling.
        train_split: Name of the training split in the dataset.
        val_split: Name of the validation split. If None, a validation set is
            carved out of the training split using val_split_size.
        val_split_size: Fraction of training data to use as validation when
            val_split is None. Ignored if val_split is provided.
        num_workers: Number of DataLoader worker processes.
    """

    audio_column: str = "audio"
    text_column: str = "text"
    sampling_rate: int = 16000
    train_split: str = "train"
    val_split: str | None = None
    val_split_size: float = 0.05
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
