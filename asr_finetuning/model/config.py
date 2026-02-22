"""Model configuration for ASR fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Configuration for the ASR model.

    Covers both base model selection and optional LoRA fine-tuning parameters.

    Args:
        model_name: HuggingFace model name or path (e.g. "unsloth/whisper-large-v3").
        use_lora: Whether to apply LoRA adapters. If False, full fine-tuning is used.
        lora_r: LoRA rank. Higher = more parameters, more capacity.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: List of module names to apply LoRA to. Common choice is:
            - ["q_proj", "v_proj"]
        language: Language name for Whisper (e.g. "English"). Set to None for auto-detection.
        task: Whisper task ("transcribe" or "translate").
        load_in_4bit: Whether to quantize to 4-bit (reduces memory usage).
    """

    model_name: str
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    language: str | None = None  # None = auto-detect language
    task: str = "transcribe"
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            ModelConfig instance.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
