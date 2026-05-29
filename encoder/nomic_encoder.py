"""
Nomic-embed-text-v1.5 interface (768-d Matryoshka output).

Set use_transformers=True to load the real Nomic model when `transformers`
is installed. Otherwise a deterministic CPU fallback fills 768-d space so
the full pipeline (projection → RSM) is runnable without a GPU or internet.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bytes_from_text(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def _hash_digest_tensor(text: str, out_dim: int, seed: int = 0) -> torch.Tensor:
    vec = torch.empty(out_dim, dtype=torch.float32)
    h = hashlib.sha256((str(seed) + "\n").encode() + _bytes_from_text(text)).digest()
    i = 0
    for j in range(out_dim):
        if i + 4 > len(h):
            h = hashlib.sha256(h + _bytes_from_text(text)).digest()
            i = 0
        b = int.from_bytes(h[i : i + 4], "big") / float(2**32) - 0.5
        vec[j] = b * 2.0
        i += 4
    return vec


class _FallbackEncoder(nn.Module):
    """768-d deterministic encoder: fixed random linear mix of bag-of-bytes features."""

    def __init__(self, encoder_dim: int = 768, n_features: int = 4096, seed: int = 42) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "proj", torch.randn(encoder_dim, n_features, generator=g) * (1.0 / math.sqrt(n_features))
        )
        self.encoder_dim = encoder_dim
        self.n_features = n_features

    def _features(self, text: str) -> torch.Tensor:
        data = _bytes_from_text(text)
        if not data:
            return torch.zeros(self.n_features, dtype=torch.float32)
        counts = torch.zeros(256, dtype=torch.float32)
        for b in data:
            counts[b] += 1.0
        counts = counts / counts.sum().clamp_min(1e-6)
        big = torch.zeros(self.n_features, dtype=torch.float32)
        big[:256] = counts
        for k in range(256, min(self.n_features, 4096), 32):
            chunk = data[k % len(data) : (k % len(data)) + 8] or data[:8]
            big[k : k + 32] = _hash_digest_tensor(
                chunk.decode("latin1", errors="replace"), 32, seed=k
            )[:32]
        return big

    def forward(self, texts: List[str]) -> torch.Tensor:
        rows = [self._features(t) for t in texts]
        x = torch.stack(rows, dim=0).to(self.proj.device)
        return x @ self.proj.T


class NomicTextEncoder(nn.Module):
    """
    Produces encoder_dim-d sentence embeddings.

    Parameters
    ----------
    use_transformers : bool
        Load the real nomic-embed-text-v1.5 via HuggingFace transformers.
    model_name : str
        HuggingFace model ID (only used when use_transformers=True).
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        use_transformers: bool = False,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self._hf_model = None
        self._hf_tokenizer = None
        self._fallback: Optional[_FallbackEncoder] = None

        if use_transformers:
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "use_transformers=True requires the `transformers` package: "
                    "pip install transformers sentencepiece"
                ) from e
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            for p in self._hf_model.parameters():
                p.requires_grad = False
            self._hf_model.eval()
        else:
            self._fallback = _FallbackEncoder(encoder_dim=encoder_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        if self._hf_model is not None and self._hf_tokenizer is not None:
            device = next(self._hf_model.parameters()).device
            tok = self._hf_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            with torch.no_grad():
                out = self._hf_model(**tok)
                last_hidden = out.last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1e-6)
                emb = summed / counts
            return emb

        assert self._fallback is not None
        return self._fallback(texts)

    def train(self, mode: bool = True) -> "NomicTextEncoder":
        super().train(mode)
        if self._hf_model is not None:
            self._hf_model.eval()
        return self
