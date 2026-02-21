"""Data configuration for ASR fine-tuning."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Schema configuration for ASR datasets.

    Describes the structure of examples in the dataset: column names, audio
    sampling rate, and the Whisper language/task tokens used during tokenization.

    Args:
        audio_column: Name of the column containing audio data.
        text_column: Name of the column containing text transcriptions.
        sampling_rate: Target sampling rate for audio resampling.
        language: Language name for Whisper (e.g. "English").
        task: Whisper task ("transcribe" or "translate").
    """

    audio_column: str = "audio"
    text_column: str = "text"
    sampling_rate: int = 16000
    language: str = "English"
    task: str = "transcribe"
