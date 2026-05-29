"""Shared tensor helpers (CPU-safe, no hard-coded CUDA)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a_n = l2_normalize(a, dim=dim, eps=eps)
    b_n = l2_normalize(b, dim=dim, eps=eps)
    return (a_n * b_n).sum(dim=dim)
