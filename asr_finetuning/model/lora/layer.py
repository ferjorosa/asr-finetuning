import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Adds trainable low-rank matrices A and B to a frozen linear layer.
    Output = (alpha / rank) * (x @ A.T @ B.T)

    A is initialized with Kaiming uniform (as in the original paper).
    B is initialized to zeros so the LoRA branch starts as a no-op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform for A — same as nn.Linear default, matches LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B = 0 so the LoRA branch is a no-op at init
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two cheap matmuls: (x @ A.T) is (*, rank), then @ B.T is (*, out_features)
        # Avoids materializing the full (out_features, in_features) weight matrix
        x = self.dropout(x)
        return self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )
