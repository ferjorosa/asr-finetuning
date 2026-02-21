"""Model factory for building Whisper models with optional LoRA."""

from typing import Any

import torch
from unsloth import FastModel
from transformers import WhisperForConditionalGeneration

from asr_finetuning.model.config import ModelConfig


def build_model(config: ModelConfig) -> tuple[torch.nn.Module, Any]:
    """Build a Whisper model from configuration.

    Handles both LoRA (PEFT) and full fine-tuning paths. Returns the model
    and processor ready for training.

    Args:
        config: Model configuration with base model name and LoRA parameters.

    Returns:
        Tuple of (model, processor). The model may be PEFT-wrapped if use_lora=True.
    """
    # Load base model with Unsloth
    base_model, processor = FastModel.from_pretrained(
        model_name=config.model_name,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
        auto_model=WhisperForConditionalGeneration,
        whisper_language=config.language,
        whisper_task=config.task,
    )

    # Apply LoRA if requested
    if config.use_lora:
        model = FastModel.get_peft_model(
            base_model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora=False,
            loftq_config=None,
            task_type=None,  # Required for Whisper
        )
    else:
        model = base_model

    # Set generation config for training (only if language is explicitly specified)
    if config.language is not None:
        model.generation_config.language = f"<|{config.language.lower()}|>"
    model.generation_config.task = config.task
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    return model, processor
