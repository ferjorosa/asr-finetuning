import torch.nn as nn

from asr_finetuning.model.lora.layer import LoRALayer
from asr_finetuning.model.lora.linear import LoRALinear


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specific modules in a model.

    Args:
        model: The model to apply LoRA to.
        rank: LoRA rank (r parameter).
        alpha: LoRA scaling factor.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.

    Returns:
        The model with LoRA applied.
    """
    target_count = 0

    for name, module in model.named_modules():
        if any(tm in name.split(".")[-1] for tm in target_modules):
            if isinstance(module, nn.Linear):
                _inject_lora(model, name, rank, alpha, dropout)
                target_count += 1

    print(f"[LoRA] Applied LoRA to {target_count} modules")
    return model


def _inject_lora(
    model: nn.Module,
    module_path: str,
    rank: int,
    alpha: int,
    dropout: float,
) -> None:
    """Inject LoRA into a specific module by path."""
    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent.get_submodule(part)

    child_name = parts[-1]
    original = parent.get_submodule(child_name)

    if not isinstance(original, nn.Linear):
        return

    lora_linear = LoRALinear(original, rank, alpha, dropout)
    setattr(parent, child_name, lora_linear)


def setup_lora_training(model: nn.Module) -> nn.Module:
    """Freeze all parameters except LoRA layers.

    Args:
        model: Model with LoRA applied.

    Returns:
        Model with only LoRA parameters trainable.
    """
    # Freeze everything first, then selectively unfreeze LoRA modules.
    # Avoids fragile string matching on parameter names.
    model.requires_grad_(False)

    trainable = 0
    frozen = 0

    for module in model.modules():
        if isinstance(module, LoRALayer):
            for param in module.parameters():
                param.requires_grad = True
                trainable += param.numel()

    for param in model.parameters():
        if not param.requires_grad:
            frozen += param.numel()

    total = trainable + frozen
    print(
        f"[LoRA] Trainable: {trainable:,} | Frozen: {frozen:,} | Ratio: {trainable / total:.4%}"
    )

    return model
