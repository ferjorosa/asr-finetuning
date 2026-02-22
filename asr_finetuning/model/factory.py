"""Model factory for building Whisper models.

Supports:
- Full fine-tuning (use_lora=False)
- LoRA from scratch (use_lora=True)
"""

from typing import Any

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_finetuning.model.config import ModelConfig
from asr_finetuning.model.lora import apply_lora_to_model, setup_lora_training
from asr_finetuning.model.lora.layer import LoRALayer


def build_model(
    model_config: ModelConfig,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Build a Whisper model from configuration.

    Handles LoRA and full fine-tuning paths. Returns the model
    and processor ready for training.

    Args:
        model_config: Model configuration with base model name and LoRA parameters.
        dtype: Model weight dtype. If None, defaults to torch.float32.

    Returns:
        Tuple of (model, processor). If use_lora=True, only LoRA parameters
        are trainable.
    """
    print(f"[build_model] dtype={dtype}")

    # -------------------------------------------------------------------------
    # 1. Load processor
    # -------------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(
        model_config.model_name,
        language=model_config.language,
        task=model_config.task,
    )

    # -------------------------------------------------------------------------
    # 2. Load base model
    # -------------------------------------------------------------------------
    if dtype is None:
        dtype = torch.float32

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_cache": False,
    }

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_name,
        **model_kwargs,
    )

    _log_dtype_summary(base_model, label="base model")

    # -------------------------------------------------------------------------
    # 3. Apply custom LoRA
    # -------------------------------------------------------------------------
    if model_config.use_lora:
        model = apply_lora_to_model(
            base_model,
            rank=model_config.lora_r,
            alpha=model_config.lora_alpha,
            dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
        )
        model = setup_lora_training(model)
        # Cast LoRA parameters to match the base model dtype so both branches
        # of LoRALinear.forward produce the same dtype and can be added.
        for module in model.modules():
            if isinstance(module, LoRALayer):
                module.to(dtype)
    else:
        model = base_model

    # -------------------------------------------------------------------------
    # 4. Gradient checkpointing
    # -------------------------------------------------------------------------
    print(f"[build_model] gradient_checkpointing={model_config.gradient_checkpointing}")
    if model_config.gradient_checkpointing:
        print("[build_model] Enabling gradient checkpointing!")
        model.gradient_checkpointing_enable()  # type: ignore[call-non-callable]
        model.train()  # type: ignore[call-non-callable]

    # -------------------------------------------------------------------------
    # 5. Whisper generation config — required for training
    # -------------------------------------------------------------------------
    if model_config.language is not None:
        model.generation_config.language = f"<|{model_config.language.lower()}|>"  # type: ignore[assignment]
    model.generation_config.task = model_config.task  # type: ignore[assignment]
    model.config.suppress_tokens = []  # type: ignore[assignment]
    model.generation_config.forced_decoder_ids = None  # type: ignore[assignment]

    return model, processor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _log_dtype_summary(model: torch.nn.Module, label: str = "model") -> None:
    """Print a summary of parameter dtypes and memory usage."""
    dtype_counts: dict[torch.dtype, int] = {}
    dtype_bytes: dict[torch.dtype, int] = {}

    for _, param in model.named_parameters():
        dt = param.dtype
        dtype_counts[dt] = dtype_counts.get(dt, 0) + param.numel()
        dtype_bytes[dt] = dtype_bytes.get(dt, 0) + param.numel() * param.element_size()

    total_mb = sum(dtype_bytes.values()) / (1024**2)
    print(f"\n[build_model] === Dtype summary: {label} ===")
    for dt, count in dtype_counts.items():
        mb = dtype_bytes[dt] / (1024**2)
        print(f"  {dt}: {count / 1e6:.2f}M params — {mb:.1f} MB")
    print(f"  Total: {total_mb:.1f} MB")

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable params: {len(trainable_params)}")
    if trainable_params:
        print(f"  First 5 trainable: {trainable_params[:5]}")
    print("[build_model] === End dtype summary ===\n")
