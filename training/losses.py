"""Contrastive and alignment losses (CPU-safe)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensor_utils import l2_normalize


def info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    anchors, positives : (B, d) L2-normalized recommended.
    Symmetric InfoNCE — same as SimCLR loss.
    """
    a = l2_normalize(anchors, dim=-1)
    p = l2_normalize(positives, dim=-1)
    logits = a @ p.T / temperature
    targets = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, targets)


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = l2_normalize(pred, dim=-1)
    tgt_n = l2_normalize(target.detach(), dim=-1)
    return 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()


def kl_divergence_logits(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    p = F.log_softmax(student_logits, dim=-1)
    q = F.softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(p, q, reduction="batchmean")
