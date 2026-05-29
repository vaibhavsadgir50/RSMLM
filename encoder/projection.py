"""Trainable encoder_dim → d_model projection with GELU and LayerNorm."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensor_utils import l2_normalize


class EncoderProjection(nn.Module):
    """
    Maps encoder-dim vectors (e.g. 768 Nomic) into unified d_model space.

    Forward: (..., encoder_dim) -> (..., d_model), L2-normalized.
    """

    def __init__(self, encoder_dim: int = 768, d_model: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.net(x), dim=-1)
