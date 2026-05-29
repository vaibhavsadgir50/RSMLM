"""Layer 2 — Resonant State Memory core (512 slots, O(1) read/write)."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rsm.contradiction import ContradictionDetector
from tensor_utils import l2_normalize


def _now_ts() -> int:
    return int(time.time() * 1000) % (2**31)


def _write_alpha(importance: float, cfg: Dict) -> float:
    if importance > 0.8:
        return float(cfg["write_alpha_high"])
    if importance > 0.5:
        return float(cfg["write_alpha_mid"])
    return float(cfg["write_alpha_low"])


class ResonantStateMemory(nn.Module):
    """
    Fixed 512 slots. Summaries are L2-normalized rows.

    read(query)  : softmax(cosine(query, slot_i)) @ summaries  -> (d_model,)
    write(input) : EMA into most resonant slot, contradiction-aware routing
    evict()      : remove lowest resonance × importance × recency slot
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        d_model = int(config["d_model"])
        n_slots = int(config["n_slots"])
        self.d_model = d_model
        self.n_slots = n_slots

        s = torch.empty(n_slots, d_model)
        nn.init.orthogonal_(s)
        self.register_buffer("summaries", l2_normalize(s, dim=-1))
        self.register_buffer("confidence", torch.full((n_slots,), 0.5))
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.long))
        self.register_buffer("importance", torch.zeros(n_slots))
        self.register_buffer("version", torch.zeros(n_slots, dtype=torch.long))
        self._pointers: List[str] = [""] * n_slots
        self._erg_derived: List[bool] = [False] * n_slots

        self.detector = ContradictionDetector(
            d_model=d_model,
            sim_threshold=float(config["contradiction_sim_threshold"]),
            score_threshold=float(config["contradiction_threshold"]),
        )

        # Learnable query projection — trained by standalone RSM training.
        # Initialized to identity-like (near-zero residual) so early reads
        # behave like raw cosine until the training signal shapes it.
        # Gradient flows: loss → retrieved → softmax(logits) → q → query_proj ✓
        # Summaries are buffers — never receive gradient from any path.
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.query_proj.weight)   # start as identity

    def pointer(self, i: int) -> str:
        return self._pointers[i]

    def set_pointer(self, i: int, key: str) -> None:
        self._pointers[i] = key

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        query : (d_model,) on any device; returns same device as summaries.

        Gradient flows through query_proj → the RSM learns how to ask questions.
        Summaries (K/V) are buffers — no gradient through slot values.
        """
        if query.shape != (self.d_model,):
            raise ValueError(f"query must be ({self.d_model},)")
        device = self.summaries.device
        q = l2_normalize(self.query_proj(query.to(device)), dim=0)
        s = l2_normalize(self.summaries, dim=-1)
        logits = torch.mv(s, q) * 8.0
        w = torch.softmax(logits, dim=0)
        out = torch.mv(s.T, w)
        return l2_normalize(out, dim=0)

    def write(
        self,
        input_vec: torch.Tensor,
        importance: float,
        erg_derived: bool = False,
        longterm_key: Optional[str] = None,
    ) -> Dict[str, object]:
        """Returns metadata dict including contradiction / ERG flags."""
        if input_vec.shape != (self.d_model,):
            raise ValueError(f"input_vec must be ({self.d_model},)")
        device = self.summaries.device
        x = l2_normalize(input_vec.detach().to(device), dim=0)
        imp = float(max(0.0, min(1.0, importance)))
        alpha = _write_alpha(imp, self.cfg)

        s = l2_normalize(self.summaries, dim=-1)
        sims = torch.mv(s, x)
        best = int(torch.argmax(sims).item())

        contradiction, conflict_idx = self.detector.detect(x, s)
        erg_flag = False
        second_slot: Optional[int] = None

        if contradiction and conflict_idx is not None:
            erg_flag = True
            sims_copy = sims.clone()
            sims_copy[conflict_idx] = -1e9
            second = int(torch.argmax(sims_copy).item())
            second_slot = second
            self._ema_write_slot(conflict_idx, x, alpha=min(alpha, 0.1), imp=imp, erg_derived=False)
            self._ema_write_slot(second, x, alpha=max(alpha, 0.3), imp=imp, erg_derived=erg_derived)
            self.version[conflict_idx] = self.version[conflict_idx] + 1
            self.version[second] = self.version[second] + 1
            if longterm_key:
                self._pointers[second] = longterm_key
        else:
            self._ema_write_slot(best, x, alpha=alpha, imp=imp, erg_derived=erg_derived)
            self.version[best] = self.version[best] + 1
            if longterm_key:
                self._pointers[best] = longterm_key

        return {
            "primary_slot": best,
            "second_slot": second_slot,
            "contradiction": contradiction,
            "erg_resolution_required": erg_flag,
        }

    def _ema_write_slot(
        self,
        idx: int,
        x: torch.Tensor,
        alpha: float,
        imp: float,
        erg_derived: bool,
    ) -> None:
        with torch.no_grad():
            cur = self.summaries[idx]
            new = (1.0 - alpha) * cur + alpha * x
            self.summaries[idx] = l2_normalize(new, dim=0)
            self.timestamps[idx] = _now_ts()
            self.importance[idx] = max(float(self.importance[idx].item()), imp)
            self.confidence[idx] = torch.clamp(
                self.confidence[idx] * 0.99 + 0.01 * imp, 0.0, 1.0
            )
            self._erg_derived[idx] = erg_derived

    def evict(self, now: Optional[int] = None) -> Optional[int]:
        """Evict lowest eviction_score = resonance × importance × recency_decay."""
        now = now or _now_ts()
        device = self.summaries.device
        s = l2_normalize(self.summaries, dim=-1)
        mean = l2_normalize(s.mean(dim=0), dim=0)
        resonance = torch.mv(s, mean)
        imp = self.importance.to(device)
        half_life = float(self.cfg.get("recency_decay_half_life", 10_000))
        age = torch.tensor(
            [max(1, now - int(self.timestamps[i].item())) for i in range(self.n_slots)],
            device=device,
            dtype=torch.float32,
        )
        recency = torch.exp(-age / max(half_life, 1.0))
        score = resonance * imp * recency

        imp_floor = float(self.cfg["eviction_importance_floor"])
        conf_floor = float(self.cfg["eviction_confidence_floor"])
        mask = (imp <= imp_floor) & (self.confidence.to(device) <= conf_floor)
        if not bool(mask.any().item()):
            return None
        score = score.masked_fill(~mask, float("inf"))
        victim = int(torch.argmin(score).item())
        with torch.no_grad():
            r = torch.empty(1, self.d_model, device=device)
            nn.init.orthogonal_(r)
            self.summaries[victim] = l2_normalize(r.squeeze(0), dim=0)
            self.confidence[victim] = 0.1
            self.importance[victim] = 0.0
            self.version[victim] = self.version[victim] + 1
            self._pointers[victim] = ""
            self._erg_derived[victim] = False
            self.timestamps[victim] = now
        return victim
