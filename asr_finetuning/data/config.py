"""Data configuration for ASR fine-tuning."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Schema configuration for ASR datasets.

    Describes the structure of examples in the dataset: column names and
    audio sampling rate.

    Args:
        audio_column: Name of the column containing audio data.
        text_column: Name of the column containing text transcriptions.
        sampling_rate: Target sampling rate for audio resampling.
    """

    audio_column: str = "audio"
    text_column: str = "text"
    sampling_rate: int = 16000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            DataConfig instance.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
