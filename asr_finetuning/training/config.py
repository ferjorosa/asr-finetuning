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
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for AdamW.
        batch_size: Batch size per device.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        precision: Lightning precision string passed to pl.Trainer.
        logging_steps: Log every N optimizer steps.
        eval_steps: Evaluate every N optimizer steps.
        save_steps: Save checkpoint every N optimizer steps.
        output_dir: Directory for outputs and checkpoints.
        run_name: Optional run name for logging.
        system_metrics_every_n_steps: Frequency for GPU/system metrics callbacks.
        resume_from_checkpoint: Optional checkpoint path to resume from.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 1
    warmup_steps: int = 5
    gradient_accumulation_steps: int = 1
    precision: Precision = "bf16-mixed"
    logging_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "outputs"
    run_name: str | None = None
    system_metrics_every_n_steps: int = 10
    resume_from_checkpoint: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
