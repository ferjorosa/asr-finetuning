"""Train an ASR model from command line."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from datasets import Audio, Dataset, load_dataset

from asr_finetuning.data.config import DataConfig
from asr_finetuning.model.config import ModelConfig
from asr_finetuning.training.config import TrainingConfig
from asr_finetuning.training.tracking import TrackioLogger
from asr_finetuning.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ASR model.")
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--training-config",
        required=True,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name (e.g. MrDragonFox/Elise).",
    )

    parser.add_argument(
        "--logger",
        choices=["trackio", "none"],
        default="trackio",
        help="Logger backend to use (default: trackio).",
    )
    return parser.parse_args()


def get_git_state() -> dict[str, str | bool]:
    """Get git SHA and dirty state for tracking."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return {"git_sha": sha, "git_dirty": bool(dirty.strip())}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_dirty": False}


def copy_run_configs(run_dir: Path, args: argparse.Namespace) -> None:
    """Copy config files to the run directory."""
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, configs_dir / Path(args.model_config).name)
    shutil.copy2(args.training_config, configs_dir / Path(args.training_config).name)
    shutil.copy2(args.data_config, configs_dir / Path(args.data_config).name)


def load_and_split_dataset(
    dataset_name: str,
    data_config: DataConfig,
) -> tuple[Dataset, Dataset]:
    """Load and prepare a HuggingFace dataset for training.

    Uses split names from DataConfig. If val_split is not defined, carves a
    validation set out of train_split using val_split_size.

    Args:
        dataset_name: HuggingFace dataset name.
        data_config: Data configuration with split names, column names, and sampling rate.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # We dont use streaming because we are aiming at small datasets (PEFT)
    dataset = load_dataset(dataset_name)

    # Cast audio column to target sampling rate
    dataset = dataset.cast_column(
        data_config.audio_column,
        Audio(sampling_rate=data_config.sampling_rate),
    )

    train = dataset[data_config.train_split]
    assert isinstance(train, Dataset)

    if data_config.val_split is not None:
        val = dataset[data_config.val_split]
        assert isinstance(val, Dataset)
        return train, val

    # No dedicated val split — carve one out of train
    split = train.train_test_split(test_size=data_config.val_split_size)
    return split["train"], split["test"]


def build_run_name(args: argparse.Namespace) -> str:
    """Build a run name from config file stems and timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = Path(args.model_config).stem
    data_name = Path(args.data_config).stem
    return f"{model_name}-{data_name}-{timestamp}"


def build_logger(
    run_name: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    args: argparse.Namespace,
) -> TrackioLogger | Literal[False]:
    """Build the trackio logger with config tracking."""
    if args.logger == "none":
        return False

    project = os.getenv("TRACKIO_PROJECT", Path(args.training_config).stem)
    return TrackioLogger(
        project=project,
        name=run_name,
        config={
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "data_config": asdict(data_config),
            "config_paths": {
                "model": args.model_config,
                "training": args.training_config,
                "data": args.data_config,
            },
            **get_git_state(),
        },
    )


def main() -> None:
    args = parse_args()

    # Load configs
    model_config = ModelConfig.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    data_config = DataConfig.from_yaml(args.data_config)

    # Build run name and directory
    run_name = training_config.run_name or build_run_name(args)
    run_dir = Path(training_config.output_base_dir) / run_name

    # Copy config files to run directory
    copy_run_configs(run_dir, args)

    # Load and split dataset
    train_dataset, val_dataset = load_and_split_dataset(
        args.dataset,
        data_config,
    )

    logger = build_logger(
        run_name=run_name,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        args=args,
    )

    # Run training
    run_training(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
