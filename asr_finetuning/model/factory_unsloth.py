"""Model factory for building Whisper models with optional LoRA."""

from typing import Any

import torch
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

    Args:
        config: Model configuration with base model name and LoRA parameters.
        dtype: Model weight dtype. So that weights are loaded in the correct
            dtype.If None, FastModel will auto-detect.

    Returns:
        Tuple of (model, processor). The model may be PEFT-wrapped if use_lora=True.
    """
    # Debug: Log the dtype being passed to FastModel
    print(f"[DEBUG] build_model: dtype parameter = {dtype}")
    print(f"[DEBUG] build_model: dtype type = {type(dtype)}")

    # Load base model with Unsloth
    base_model, processor = FastModel.from_pretrained(
        model_name=config.model_name,
        dtype=dtype,
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

    # Debug: Check gradient checkpointing status
    print("\n[DEBUG] === Gradient Checkpointing Status ===")
    if hasattr(model, "gradient_checkpointing_enable"):
        print("[DEBUG] Model has gradient_checkpointing_enable method")
    if hasattr(model, "is_gradient_checkpointing"):
        print(
            f"[DEBUG] Model is_gradient_checkpointing: {model.is_gradient_checkpointing}"
        )

    # Debug: Print model structure to find decoder
    print(f"[DEBUG] Model type: {type(model)}")
    print(
        f"[DEBUG] Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}"
    )
    if hasattr(model, "model"):
        base = model.model
        print(f"[DEBUG] base_model.model type: {type(base)}")
        print(
            f"[DEBUG] base_model.model attributes: {[a for a in dir(base) if not a.startswith('_')][:20]}"
        )

    # Check if encoder/decoder have gradient checkpointing
    # Structure: PeftModel -> WhisperForConditionalGeneration -> WhisperModel -> encoder/decoder
    if hasattr(model, "model"):
        whisper_for_cg = model.model  # WhisperForConditionalGeneration
        if hasattr(whisper_for_cg, "model"):
            whisper_model = whisper_for_cg.model  # WhisperModel
            if hasattr(whisper_model, "encoder"):
                enc = whisper_model.encoder
                if hasattr(enc, "gradient_checkpointing"):
                    print(
                        f"[DEBUG] Encoder gradient_checkpointing: {enc.gradient_checkpointing}"
                    )
                else:
                    print(
                        "[DEBUG] WARNING: Encoder has no gradient_checkpointing attribute!"
                    )
            if hasattr(whisper_model, "decoder"):
                dec = whisper_model.decoder
                if hasattr(dec, "gradient_checkpointing"):
                    print(
                        f"[DEBUG] Decoder gradient_checkpointing: {dec.gradient_checkpointing}"
                    )
                else:
                    print(
                        "[DEBUG] WARNING: Decoder has no gradient_checkpointing attribute!"
                    )
            else:
                print("[DEBUG] WARNING: WhisperModel has no decoder!")

    # Check all modules for gradient_checkpointing attribute
    gc_modules = []
    encoder_gc_modules = []
    decoder_gc_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            gc_modules.append((name, module.gradient_checkpointing))
            if "encoder" in name:
                encoder_gc_modules.append((name, module.gradient_checkpointing))
            elif "decoder" in name:
                decoder_gc_modules.append((name, module.gradient_checkpointing))
    if gc_modules:
        print("[DEBUG] Modules with gradient_checkpointing attribute:")
        for name, gc_val in gc_modules[:10]:  # Show first 10
            print(f"[DEBUG]   {name}: {gc_val}")
        if len(gc_modules) > 10:
            print(f"[DEBUG]   ... and {len(gc_modules) - 10} more")
        print(
            f"[DEBUG] Total: {len(gc_modules)} modules ({len(encoder_gc_modules)} encoder, {len(decoder_gc_modules)} decoder)"
        )
    else:
        print("[DEBUG] WARNING: No modules have gradient_checkpointing attribute!")
    print("[DEBUG] === End Gradient Checkpointing Status ===\n")

    # Debug: Comprehensive dtype and memory analysis
    print("\n[DEBUG] === Model Memory Analysis ===")

    # Count params by dtype
    dtype_counts = {}
    dtype_memory = {}
    for name, param in model.named_parameters():
        dt = param.dtype
        if dt not in dtype_counts:
            dtype_counts[dt] = 0
            dtype_memory[dt] = 0
        dtype_counts[dt] += param.numel()
        dtype_memory[dt] += param.numel() * param.element_size()

    print("[DEBUG] Parameters by dtype:")
    for dt, count in dtype_counts.items():
        mem_mb = dtype_memory[dt] / (1024 * 1024)
        print(f"[DEBUG]   {dt}: {count / 1e6:.2f}M params, {mem_mb:.2f} MB")

    # Total memory estimate
    total_mem = sum(dtype_memory.values()) / (1024 * 1024)
    print(f"[DEBUG] Total params memory: {total_mem:.2f} MB")

    # Check for any FP32 params (potential issue)
    fp32_params = [
        name for name, p in model.named_parameters() if p.dtype == torch.float32
    ]
    if fp32_params:
        print(f"[DEBUG] WARNING: {len(fp32_params)} params in FP32!")
        print(f"[DEBUG] First 5 FP32 params: {fp32_params[:5]}")

    print("[DEBUG] === End Memory Analysis ===\n")

    # Set generation config for training (only if language is explicitly specified)
    if config.language is not None:
        model.generation_config.language = f"<|{config.language.lower()}|>"
    model.generation_config.task = config.task
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    return model, processor
