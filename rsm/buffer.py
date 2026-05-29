"""Layer 1 — importance-weighted episodic buffer (soft capacity)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from rsm.importance_scorer import ImportanceScorer
from tensor_utils import l2_normalize


def _now_ts() -> int:
    return int(time.time() * 1000) % (2**31)


@dataclass
class BufferEntry:
    vector: torch.Tensor
    text: str
    importance: float
    timestamp: int = field(default_factory=_now_ts)


class ImportanceBuffer:
    """
    Importance-weighted episodic buffer. Evicts lowest-importance entry (not oldest).

    read ranks by cosine_similarity(query, v) * importance.
    """

    def __init__(self, capacity: int = 50, d_model: int = 1024) -> None:
        self.capacity = capacity
        self.d_model = d_model
        self._entries: List[BufferEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, vector: torch.Tensor, text: str, importance: float | None = None) -> None:
        if vector.shape != (self.d_model,):
            raise ValueError(f"vector must be ({self.d_model},), got {tuple(vector.shape)}")
        v = l2_normalize(vector.detach(), dim=0).cpu()
        imp = float(ImportanceScorer.score(text) if importance is None else importance)
        imp = float(max(0.0, min(1.0, imp)))
        entry = BufferEntry(vector=v, text=text, importance=imp, timestamp=_now_ts())

        if len(self._entries) >= self.capacity:
            idx = min(range(len(self._entries)), key=lambda i: self._entries[i].importance)
            self._entries.pop(idx)
        self._entries.append(entry)

    def read(self, query_vector: torch.Tensor, top_k: int = 5) -> List[BufferEntry]:
        if query_vector.shape != (self.d_model,):
            raise ValueError(f"query_vector must be ({self.d_model},)")
        if not self._entries:
            return []
        q = l2_normalize(query_vector.detach(), dim=0).cpu()
        scored: List[Tuple[float, BufferEntry]] = []
        for e in self._entries:
            sim = float((q * e.vector).sum())
            scored.append((sim * e.importance, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]
