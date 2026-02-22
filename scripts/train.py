"""Train an ASR model from command line."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from datasets import Audio, Dataset
from datasets import load_dataset

from asr_finetuning.data.config import DataConfig
from asr_finetuning.model.config import ModelConfig
from asr_finetuning.training.config import TrainingConfig
from asr_finetuning.training.tracking import TrackioLogger
from asr_finetuning.training.trainer import run as run_training


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
    parser.add_argument(
        "--trackio-project",
        default=None,
        help="Trackio project name (defaults to TRACKIO_PROJECT or 'asr-finetuning').",
    )
    return parser.parse_args()


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


def build_logger(
    logger_name: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    trackio_project: str | None,
):
    if logger_name == "none":
        return False

    project = trackio_project or os.getenv("TRACKIO_PROJECT", "asr-finetuning")
    return TrackioLogger(
        project=project,
        name=training_config.run_name,
        config={
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "data_config": asdict(data_config),
        },
    )


def main() -> None:
    args = parse_args()

    # Load configs
    model_config = ModelConfig.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    data_config = DataConfig.from_yaml(args.data_config)

    # Load and split dataset
    train_dataset, val_dataset = load_and_split_dataset(
        args.dataset,
        data_config,
    )
    logger = build_logger(
        logger_name=args.logger,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        trackio_project=args.trackio_project,
    )

    # Run training
    run_training(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
    )


if __name__ == "__main__":
    main()
