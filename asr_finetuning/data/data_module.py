"""Lightning data module for ASR fine-tuning."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from asr_finetuning.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from asr_finetuning.data.config import DataConfig
from asr_finetuning.data.dataset import ASRDataset


class ASRDataModule(pl.LightningDataModule):
    """LightningDataModule for ASR fine-tuning.

    Accepts pre-loaded datasets, decoupling data loading from data serving.
    Loading, splitting, and any source-specific preparation (HuggingFace Hub,
    local files, custom formats) is the responsibility of the caller.

    Args:
        train_dataset: Training split (any object that is indexable and has a length).
        val_dataset: Validation split.
        config: Data configuration (column names, sampling rate).
        processor: WhisperProcessor for feature extraction and tokenization.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader worker processes.
    """

    def __init__(
        self,
        train_dataset: Any,
        val_dataset: Any,
        config: DataConfig,
        processor: Any,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataset = ASRDataset(
            hf_dataset=train_dataset, processor=processor, config=config
        )
        self._val_dataset = ASRDataset(
            hf_dataset=val_dataset, processor=processor, config=config
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            num_workers=self.num_workers,
            shuffle=False,
        )
