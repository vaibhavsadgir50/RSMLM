"""
RSMemory — orchestrates all 3 tiers.

  Layer 1: ImportanceBuffer  — fast episodic buffer, evicts by importance
  Layer 2: ResonantStateMemory — 512 fixed slots, O(1) read/write
  Layer 3: LongTermStore     — versioned topic chains, async extract
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from encoder.nomic_encoder import NomicTextEncoder
from encoder.projection import EncoderProjection
from rsm.buffer import ImportanceBuffer
from rsm.core import ResonantStateMemory
from rsm.importance_scorer import ImportanceScorer
from rsm.longterm import LongTermStore
from tensor_utils import l2_normalize


class RSMemory(nn.Module):
    """
    End-to-end 3-tier RSM:
      text → encoder (768-d) → projection (d_model) → Layer1 buffer
                                                     → Layer2 slots (read/write)
                                                     → Layer3 longterm (async)

    If encode_fn is provided it replaces the built-in encoder + projection.
    Otherwise NomicTextEncoder (frozen) + EncoderProjection (trainable) are used.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        encode_fn: Optional[Callable[[str], torch.Tensor]] = None,
        *,
        use_transformers: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.d_model = int(config["d_model"])
        self.encode_fn = encode_fn

        # Layer 1
        self.buffer = ImportanceBuffer(
            capacity=int(config["buffer_capacity"]),
            d_model=self.d_model,
        )
        # Layer 2
        self.rsm = ResonantStateMemory(config)
        # Layer 3
        self.longterm = LongTermStore(d_model=self.d_model)

        # Built-in encoder (only when no encode_fn supplied)
        self.text_encoder: Optional[NomicTextEncoder] = None
        self.text_projection: Optional[EncoderProjection] = None
        if encode_fn is None:
            edim = int(config["encoder_dim"])
            self.text_encoder = NomicTextEncoder(
                encoder_dim=edim, use_transformers=use_transformers
            )
            self.text_projection = EncoderProjection(
                encoder_dim=edim, d_model=self.d_model
            )
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Aliases for external tooling
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> Optional[NomicTextEncoder]:
        return self.text_encoder

    @property
    def projection(self) -> Optional[EncoderProjection]:
        return self.text_projection

    @property
    def rsm_core(self) -> ResonantStateMemory:
        return self.rsm

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def encode(self, text: str) -> torch.Tensor:
        if self.encode_fn is not None:
            v = self.encode_fn(text)
            if v.shape != (self.d_model,):
                raise ValueError(f"encode_fn must return ({self.d_model},), got {tuple(v.shape)}")
            return v

        if self.text_encoder is None or self.text_projection is None:
            raise RuntimeError("encode_fn not set and built-in encoder not initialized")

        device = self.rsm.summaries.device
        self.text_encoder.to(device)
        self.text_projection.to(device)
        with torch.no_grad():
            e = self.text_encoder([text])
        e = e.to(device)
        z = self.text_projection(e[0])
        return z

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, query_text: str, input_text: str) -> torch.Tensor:
        """
        Full 3-tier pass. Returns memory_context (d_model,), L2-normalized.
        """
        query_vec = self.encode(query_text)
        input_vec = self.encode(input_text)
        return self.forward_vectors(query_vec, input_text, input_vec)

    def forward_vectors(
        self,
        query_vec: torch.Tensor,
        input_text: str,
        input_vec: torch.Tensor,
    ) -> torch.Tensor:
        device = self.rsm.summaries.device
        query_vec = query_vec.to(device)
        input_vec = input_vec.to(device)

        importance = ImportanceScorer.score(input_text)

        # Layer 1 write
        self.buffer.write(input_vec, input_text, importance)

        # Layer 2 read then write
        memory_context = self.rsm.read(query_vec)
        self.rsm.write(input_vec, importance=importance, erg_derived=False)

        # Layer 3 async extract for high-importance inputs
        thr = float(self.cfg["importance_threshold"])
        if importance > thr:
            def _attach(key: str) -> None:
                if not key:
                    return
                s = l2_normalize(self.rsm.summaries, dim=-1)
                v = l2_normalize(input_vec, dim=0)
                sims = torch.mv(s, v)
                idx = int(torch.argmax(sims).item())
                self.rsm.set_pointer(idx, key)

            self.longterm.schedule_extract(input_vec, input_text, importance, on_complete=_attach)

        return memory_context

    # ------------------------------------------------------------------
    # ERG feedback path
    # ------------------------------------------------------------------

    def erg_feedback_write(self, conclusion: torch.Tensor, confidence: float) -> Dict[str, object]:
        """Called from ERG when conclusion_confidence > threshold."""
        if conclusion.shape != (self.d_model,):
            raise ValueError(f"conclusion must be ({self.d_model},)")
        imp = float(max(0.0, min(1.0, confidence)))
        return self.rsm.write(
            conclusion.to(self.rsm.summaries.device),
            importance=imp,
            erg_derived=True,
        )

    # ------------------------------------------------------------------
    # Deep-copy safety
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k in ("_extract_thread", "_thread_lock", "_lock"):
                setattr(new_obj, k, None)
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj
