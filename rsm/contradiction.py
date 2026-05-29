"""Contradiction detection before RSM slot writes (vector-space heuristic)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tensor_utils import l2_normalize


class ContradictionDetector(nn.Module):
    """
    detect(new_vector, existing_slot_vectors)
        -> (contradiction: bool, conflicting_slot_index: Optional[int])

    High cosine similarity plus opposing sign along a fixed random probe axis
    yields a contradiction score — no extra dependencies required.
    """

    def __init__(
        self,
        d_model: int = 1024,
        sim_threshold: float = 0.85,
        score_threshold: float = 0.7,
        seed: int = 123,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        probe = torch.randn(d_model, generator=g)
        self.register_buffer("probe", l2_normalize(probe, dim=0))
        self.d_model = d_model
        self.sim_threshold = sim_threshold
        self.score_threshold = score_threshold

    def detect(
        self,
        new_vector: torch.Tensor,
        existing_slots: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple[bool, Optional[int]]:
        """
        new_vector    : (d_model,)
        existing_slots: (n_slots, d_model)
        """
        if new_vector.dim() != 1:
            raise ValueError("new_vector must be 1-D (d_model,)")
        if existing_slots.dim() != 2 or existing_slots.shape[-1] != new_vector.shape[0]:
            raise ValueError("existing_slots must be (n_slots, d_model)")

        device = new_vector.device
        probe = self.probe.to(device)
        nv = l2_normalize(new_vector, dim=0)
        slots = l2_normalize(existing_slots, dim=-1)

        sims = torch.mv(slots, nv)
        k = min(top_k, sims.numel())
        top_vals, top_idx = torch.topk(sims, k=k)

        pn = (nv * probe).sum()
        for rank in range(k):
            cos = top_vals[rank].item()
            if cos <= self.sim_threshold:
                continue
            j = int(top_idx[rank].item())
            sj = slots[j]
            ps = (sj * probe).sum()
            opposition = torch.sigmoid(-5.0 * pn * ps)
            contradiction_score = cos * opposition
            if contradiction_score.item() > self.score_threshold:
                return True, j

        return False, None
