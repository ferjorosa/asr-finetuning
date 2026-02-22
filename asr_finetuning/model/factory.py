"""Model factory for building Whisper models with optional LoRA.

Uses Unsloth's FastModel for efficient base model loading (handles dtype,
quantization, and Whisper-specific setup), but standard PEFT for LoRA
application to ensure full PyTorch Lightning compatibility and correct
gradient checkpointing behavior.
"""

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration
from unsloth import FastModel

from asr_finetuning.model.config import ModelConfig


def build_model(
    config: ModelConfig,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Build a Whisper model from configuration.

    Handles both LoRA (PEFT) and full fine-tuning paths. Returns the model
    and processor ready for training.

    Uses Unsloth for base model loading (dtype handling, quantization,
    Whisper-specific configuration) but standard HuggingFace PEFT for LoRA
    to ensure compatibility with PyTorch Lightning's training loop and correct
    gradient checkpointing behavior.

    Args:
        config: Model configuration with base model name and LoRA parameters.
        dtype: Model weight dtype. If None, FastModel will auto-detect.

    Returns:
        Tuple of (model, processor). The model may be PEFT-wrapped if
        config.use_lora=True.
    """
    print(f"[build_model] dtype={dtype}")

    # -------------------------------------------------------------------------
    # 1. Load base model via Unsloth
    #    FastModel handles: dtype auto-detection, 4-bit quantization, and
    #    Whisper-specific tokenizer/processor setup.
    #    We do NOT use FastModel.get_peft_model — standard PEFT is used instead
    #    to avoid Unsloth's custom GC patches that conflict with Lightning.
    # -------------------------------------------------------------------------
    base_model, processor = FastModel.from_pretrained(
        model_name=config.model_name,
        dtype=dtype,
        load_in_4bit=config.load_in_4bit,
        auto_model=WhisperForConditionalGeneration,
        whisper_language=config.language,
        whisper_task=config.task,
    )

    _log_dtype_summary(base_model, label="base model")

    # -------------------------------------------------------------------------
    # 2. Enable standard PyTorch gradient checkpointing on the base model
    #    BEFORE wrapping with PEFT. This uses HuggingFace's standard GC
    #    implementation which is fully compatible with Lightning.
    #
    #    We call this on the base model (not the PEFT wrapper) because PEFT
    #    wraps the model and calling gradient_checkpointing_enable afterwards
    #    may not reach the inner model's encoder/decoder correctly.
    # -------------------------------------------------------------------------
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    _verify_gradient_checkpointing(base_model, label="base model (after GC enable)")

    # -------------------------------------------------------------------------
    # 3. Apply LoRA via standard PEFT
    # -------------------------------------------------------------------------
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            # task_type must be None for Whisper — using TaskType.SEQ_2_SEQ_LM
            # breaks generation because PEFT wraps the forward in a way that
            # conflicts with Whisper's encoder-decoder cross-attention setup.
            task_type=None,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model

    _verify_gradient_checkpointing(model, label="final model (after PEFT wrap)")

    # -------------------------------------------------------------------------
    # 4. Disable KV cache for training
    #    When use_cache=True (default), Whisper's decoder caches KV states
    #    which prevents gradient checkpointing from recomputing them during
    #    the backward pass, defeating memory savings.
    # -------------------------------------------------------------------------
    model.config.use_cache = False  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # 5. Whisper generation config — required for training
    # -------------------------------------------------------------------------
    if config.language is not None:
        model.generation_config.language = f"<|{config.language.lower()}|>"
    model.generation_config.task = config.task
    model.config.suppress_tokens = []  # type: ignore[assignment]
    model.generation_config.forced_decoder_ids = None

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

    fp32_params = [n for n, p in model.named_parameters() if p.dtype == torch.float32]
    if fp32_params:
        print(f"  WARNING: {len(fp32_params)} params still in FP32!")
        print(f"  First 5: {fp32_params[:5]}")
    print("[build_model] === End dtype summary ===\n")


def _verify_gradient_checkpointing(
    model: torch.nn.Module, label: str = "model"
) -> None:
    """Check and log gradient checkpointing status across encoder and decoder."""
    print(f"\n[build_model] === GC status: {label} ===")

    # Walk the module tree looking for gradient_checkpointing attributes
    gc_on: list[str] = []
    gc_off: list[str] = []

    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            if module.gradient_checkpointing:
                gc_on.append(name or "<root>")
            else:
                gc_off.append(name or "<root>")

    if gc_on:
        print(f"  GC enabled on {len(gc_on)} module(s):")
        for n in gc_on[:10]:
            print(f"    ✓ {n}")
        if len(gc_on) > 10:
            print(f"    ... and {len(gc_on) - 10} more")
    else:
        print("  WARNING: GC is not enabled on any module!")

    if gc_off:
        print(f"  GC disabled on {len(gc_off)} module(s) (first 5):")
        for n in gc_off[:5]:
            print(f"    ✗ {n}")

    print("[build_model] === End GC status ===\n")
