"""Training configuration for ASR fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import yaml


Precision: TypeAlias = Literal[
    32,
    16,
    "bf16-mixed",
    "16-mixed",
    "32-true",
    "bf16-true",
]


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    The precision field is passed directly to pl.Trainer(precision=...).
    Use "bf16-mixed" on modern GPUs (Ampere+), "16-mixed" on older ones.

    Args:
        precision: Lightning precision string passed to pl.Trainer.
        weight_decay: Weight decay for AdamW.
        learning_rate: Learning rate for the optimizer.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        batch_size: Batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        num_epochs: Number of training epochs.
        val_every_n_steps: Run validation every N optimizer steps.
        system_metrics_every_n_steps: Frequency for GPU/system metrics and logging.
        save_every_n_steps: Save checkpoint every N optimizer steps.
        output_dir: Directory for outputs and checkpoints.
        resume_from_checkpoint: Optional checkpoint path to resume from.
        run_name: Optional run name for logging.
    """

    # Precision
    precision: Precision = "bf16-mixed"

    # Optimizer and training stability
    weight_decay: float = 0.01

    # Learning-rate schedule
    learning_rate: float = 1e-4
    warmup_steps: int = 5

    # Tokens, steps, and batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1

    # Validation and logging
    val_every_n_steps: int = 100
    system_metrics_every_n_steps: int = 10

    # Checkpointing
    save_every_n_steps: int = 500
    output_dir: str = "outputs"
    resume_from_checkpoint: str | None = None

    # Logging
    run_name: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
