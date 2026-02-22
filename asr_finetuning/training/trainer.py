"""Trainer orchestrator for ASR fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from asr_finetuning.data.config import DataConfig
from asr_finetuning.data.data_module import ASRDataModule
from asr_finetuning.model.config import ModelConfig
from asr_finetuning.model.factory import build_model
from asr_finetuning.training.callbacks import GpuStatsMonitor
from asr_finetuning.training.config import TrainingConfig
from asr_finetuning.training.module import WhisperModule


def run_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    train_dataset: Any,
    val_dataset: Any,
    run_dir: Path,
    logger: Any = False,
) -> None:
    """Run training for an ASR model.

    This is the main orchestration function. It builds the model, creates the
    Lightning module and data module, sets up the Trainer, and runs training.

    Args:
        model_config: Model configuration (architecture, LoRA params, etc).
        training_config: Training configuration (lr, batch size, epochs, etc).
        data_config: Data configuration (column names, sampling rate).
        train_dataset: Pre-loaded training dataset.
        val_dataset: Pre-loaded validation dataset.
        logger: Lightning logger instance or False to disable logging.
    """
    # Derive model dtype from training precision.
    # The dtype must be passed to BOTH the model (for weights) and data module
    # (for input features). Lightning's precision in the Trainer only autocasts
    # the forward pass, it does NOT cast model weights or input data automatically.
    _precision_to_dtype: dict[str | int, torch.dtype] = {
        "bf16-mixed": torch.bfloat16,
        "bf16-true": torch.bfloat16,
        "16-mixed": torch.float16,
        16: torch.float16,
        "32-true": torch.float32,
        32: torch.float32,
    }
    model_dtype = _precision_to_dtype.get(training_config.precision)

    # Build model (returns model and processor)
    model, processor = build_model(model_config, dtype=model_dtype)

    # Create Lightning module
    module = WhisperModule(
        model=model,
        processor=processor,
        config=training_config,
    )

    # Create data module
    data_module = ASRDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=data_config,
        processor=processor,
        batch_size=training_config.batch_size,
        dtype=model_dtype,  # Cast inputs to match model precision
    )

    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            every_n_train_steps=training_config.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        GpuStatsMonitor(log_every_n_steps=training_config.system_metrics_every_n_steps),
    ]

    # val_check_interval is in batches; convert from optimizer steps
    val_check_interval = (
        training_config.val_every_n_steps * training_config.gradient_accumulation_steps
    )

    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator="auto",
        devices="auto",
        precision=training_config.precision,
        max_epochs=training_config.num_epochs,
        val_check_interval=val_check_interval,
        limit_val_batches=training_config.limit_val_batches,
        log_every_n_steps=training_config.system_metrics_every_n_steps,
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        gradient_clip_val=training_config.grad_clip_norm or None,
        callbacks=callbacks,
        logger=logger,
    )

    # Run training
    trainer.fit(
        module, datamodule=data_module, ckpt_path=training_config.resume_from_checkpoint
    )
