"""Layer 3 — versioned long-term store (pointer in RSM slot, body here)."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from tensor_utils import l2_normalize


def _now_ts() -> int:
    return int(time.time() * 1000) % (2**31)


@dataclass
class LongTermEntry:
    vector: torch.Tensor
    text: str
    version: int
    timestamp: int
    importance: float
    superseded: bool = False


class LongTermStore:
    """
    storage: key -> versioned entry chain (only non-superseded entries are queried).

    extract() can be scheduled asynchronously so the RSM hot path is not blocked.
    """

    def __init__(self, d_model: int = 1024, topic_sim_threshold: float = 0.9) -> None:
        self.d_model = d_model
        self.topic_sim_threshold = topic_sim_threshold
        self._storage: Dict[str, List[LongTermEntry]] = {}
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        new_store = LongTermStore(
            d_model=self.d_model,
            topic_sim_threshold=self.topic_sim_threshold,
        )
        with self._lock:
            new_store._storage = copy.deepcopy(self._storage, memo)
        return new_store

    def _new_key(self) -> str:
        return str(uuid.uuid4())

    def _find_topic_key(self, vector: torch.Tensor) -> Optional[str]:
        v = l2_normalize(vector.detach().cpu(), dim=0)
        best_k: Optional[str] = None
        best_sim = self.topic_sim_threshold
        for k, chain in self._storage.items():
            active = [e for e in chain if not e.superseded]
            if not active:
                continue
            ev = l2_normalize(active[-1].vector, dim=0)
            sim = float((v * ev).sum())
            if sim >= best_sim:
                best_sim = sim
                best_k = k
        return best_k

    def extract_sync(self, vector: torch.Tensor, text: str, importance: float) -> str:
        """Synchronous extract; returns storage key used. Only stores if importance > 0.85."""
        if vector.shape != (self.d_model,):
            raise ValueError(f"vector must be ({self.d_model},)")
        imp = float(max(0.0, min(1.0, importance)))
        if imp <= 0.85:
            return ""
        vec = l2_normalize(vector.detach().cpu(), dim=0)
        with self._lock:
            k = self._find_topic_key(vec)
            if k is None:
                k = self._new_key()
                self._storage[k] = []
                ver = 1
            else:
                chain = self._storage[k]
                for e in chain:
                    if not e.superseded:
                        e.superseded = True
                ver = max((e.version for e in chain), default=0) + 1
            ent = LongTermEntry(
                vector=vec.clone(),
                text=text,
                version=ver,
                timestamp=_now_ts(),
                importance=imp,
                superseded=False,
            )
            self._storage[k].append(ent)
        return k

    def schedule_extract(
        self,
        vector: torch.Tensor,
        text: str,
        importance: float,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Non-blocking: clones tensor and processes in a background thread."""
        vec_cpu = vector.detach().cpu().clone()
        text_copy = str(text)
        imp = float(importance)

        def job() -> None:
            key = self.extract_sync(vec_cpu, text_copy, imp)
            if on_complete is not None:
                on_complete(key)

        t = threading.Thread(target=job, daemon=True)
        t.start()

    def query(self, query_vector: torch.Tensor, top_k: int = 3) -> List[LongTermEntry]:
        if query_vector.shape != (self.d_model,):
            raise ValueError(f"query_vector must be ({self.d_model},)")
        q = l2_normalize(query_vector.detach().cpu(), dim=0)
        scored: List[Tuple[float, LongTermEntry]] = []
        with self._lock:
            for chain in self._storage.values():
                for e in chain:
                    if e.superseded:
                        continue
                    ev = l2_normalize(e.vector, dim=0)
                    sim = float((q * ev).sum())
                    scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get_chain(self, key: str) -> List[LongTermEntry]:
        with self._lock:
            return list(self._storage.get(key, []))

    def num_topics(self) -> int:
        with self._lock:
            return len(self._storage)
