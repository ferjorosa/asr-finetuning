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

Scheduler: TypeAlias = Literal["cosine", "none"]

Optimizer: TypeAlias = Literal["adamw"]


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    The precision field is passed directly to pl.Trainer(precision=...).
    Use "bf16-mixed" on modern GPUs (Ampere+), "16-mixed" on older ones.

    Args:
        precision: Lightning precision string passed to pl.Trainer.
        optimizer: Optimizer variant. "adamw" uses standard PyTorch AdamW.
        weight_decay: Weight decay for AdamW.
        betas: AdamW beta coefficients (beta1, beta2).
        eps: AdamW epsilon for numerical stability.
        grad_clip_norm: Max gradient norm for clipping. 0 disables clipping.
        learning_rate: Peak learning rate.
        warmup_steps: Number of linear warmup optimizer steps.
        scheduler: LR schedule type ("cosine" or "none").
        min_lr: Minimum LR at the end of cosine decay. Ignored when scheduler="none".
        batch_size: Batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        num_epochs: Number of training epochs.
        val_every_n_steps: Run validation every N optimizer steps.
        limit_val_batches: Fraction (0.0-1.0) or number of validation batches to run.
            Set to 0.0 to disable validation entirely.
        system_metrics_every_n_steps: Frequency for GPU/system metrics and logging.
        save_every_n_steps: Save checkpoint every N optimizer steps.
        output_base_dir: Base directory for outputs and checkpoints.
        resume_from_checkpoint: Optional checkpoint path to resume from.
        run_name: Optional run name for logging.
    """

    # Precision
    precision: Precision = "bf16-mixed"

    # Optimizer
    optimizer: Optimizer = "adamw"
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Learning-rate schedule
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    scheduler: Scheduler = "cosine"
    min_lr: float = 1e-5

    # Batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1

    # Validation and logging
    val_every_n_steps: int = 100
    limit_val_batches: float | int = (
        1.0  # Fraction or number of val batches; 0.0 disables validation
    )
    system_metrics_every_n_steps: int = 10

    # Checkpointing
    save_every_n_steps: int = 500
    output_base_dir: str = "outputs"
    resume_from_checkpoint: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # YAML loads as list
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be non-negative")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.val_every_n_steps <= 0:
            raise ValueError("val_every_n_steps must be positive")
        if self.system_metrics_every_n_steps <= 0:
            raise ValueError("system_metrics_every_n_steps must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
