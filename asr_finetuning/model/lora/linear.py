import torch
import torch.nn as nn

from asr_finetuning.model.lora.layer import LoRALayer


class LoRALinear(nn.Module):
    """Linear layer with LoRA applied in parallel.

    Wraps an existing Linear layer and adds LoRA adaptation.
    Forward: y = W @ x + lora(x)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.linear.requires_grad_(False)

        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    def extra_repr(self) -> str:
        return f"rank={self.lora.rank}, alpha={self.lora.alpha}"
