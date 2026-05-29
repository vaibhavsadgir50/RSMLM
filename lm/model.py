"""
RSMLM — Language model with no KV cache.

Design
------
The model processes text in fixed-size chunks (block_size tokens).
For each chunk it does two things per layer:

  1. Causal self-attention  — attends to the CURRENT chunk only (O(T²), T is fixed).
                              No KV cache ever needs to grow. After the chunk is done,
                              those activations are thrown away.

  2. RSM cross-attention    — Q comes from the current hidden state;
                              K/V come from the RSM slot matrix (n_slots × d_model),
                              which is FIXED-SIZE regardless of how much text has been read.
                              The RSM is the long-range memory. The model learns to query it.

After each chunk the mean hidden state is written back into RSM via EMA.
At the start of each document RSM is reset.

The RSM state is always detached from the compute graph — no gradient flows through
RSM buffers. Gradients flow through the model's cross-attention projections (q, k, v),
which is how the model learns to read RSM effectively.

Public interface
----------------
  model = RSMLM(LMConfig(...))
  rsm   = model.build_rsm()          # creates a fresh ResonantStateMemory

  # inference loop (no KV cache needed):
  rsm.evict()                         # optional maintenance
  logits, hidden = model(idx, rsm)   # idx: (B, T)
  model.write_to_rsm(hidden, rsm)    # update memory with what we just saw

  # training loop:
  # same as above, but call optimizer.step() after loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from rsm.core import ResonantStateMemory
from tensor_utils import l2_normalize


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMConfig:
    vocab_size:  int   = 65          # set by dataset
    d_model:     int   = 256         # LM hidden dim
    n_layer:     int   = 4           # number of transformer blocks
    n_head:      int   = 8           # attention heads (d_model must divide n_head)
    ffn_dim:     int   = 1024        # FFN hidden size (4 × d_model default)
    block_size:  int   = 128         # max tokens per chunk (replaces context window)
    dropout:     float = 0.1
    # RSM config
    rsm_d_model: int   = 256         # RSM slot dimension (may differ from d_model)
    rsm_n_slots: int   = 512         # number of RSM memory slots
    rsm_beta:    float = 8.0         # RSM read attention temperature

    # Presets -----------------------------------------------------------------
    @classmethod
    def tiny(cls, vocab_size: int = 65) -> "LMConfig":
        """CPU-trainable in minutes. ~0.5M params."""
        return cls(vocab_size=vocab_size, d_model=128, n_layer=2,
                   n_head=4, ffn_dim=512, block_size=64,
                   rsm_d_model=128, rsm_n_slots=128)

    @classmethod
    def small(cls, vocab_size: int = 65) -> "LMConfig":
        """CPU-trainable in ~30 min, GPU in ~5 min. ~3M params."""
        return cls(vocab_size=vocab_size, d_model=256, n_layer=4,
                   n_head=8, ffn_dim=1024, block_size=128,
                   rsm_d_model=256, rsm_n_slots=256)

    @classmethod
    def base(cls, vocab_size: int = 65) -> "LMConfig":
        """GPU recommended. ~15M params."""
        return cls(vocab_size=vocab_size, d_model=512, n_layer=6,
                   n_head=8, ffn_dim=2048, block_size=256,
                   rsm_d_model=512, rsm_n_slots=512)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


class CausalSelfAttn(nn.Module):
    """
    Standard causal MHA over the current chunk only.

    T is always <= block_size (fixed). No KV cache is needed or stored —
    once this chunk's forward pass is done, the activations are freed.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head  = n_head
        self.head_dim = d_model // n_head
        self.scale   = 1.0 / math.sqrt(self.head_dim)
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)                             # (B, T, nh, hd)
        q = q.transpose(1, 2)                               # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale        # (B, nh, T, T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class RSMCrossAttn(nn.Module):
    """
    Cross-attention: Q from current token hidden states, K/V from RSM slots.

    RSM slots are DETACHED — no gradient flows into RSM buffers.
    Gradient flows through q_proj, k_proj, v_proj (model parameters),
    so the model learns how to query and interpret RSM.

    If lm_d != rsm_d, a slot_proj Linear maps RSM dim to LM dim first.
    """

    def __init__(self, d_model: int, n_head: int, rsm_d_model: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head   = n_head
        self.head_dim = d_model // n_head
        self.scale    = 1.0 / math.sqrt(self.head_dim)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(rsm_d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(rsm_d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rsm_slots: torch.Tensor) -> torch.Tensor:
        """
        x         : (B, T, d_model)
        rsm_slots : (n_slots, rsm_d_model) — detached RSM summaries
        returns   : (B, T, d_model)
        """
        B, T, C = x.shape
        S = rsm_slots.shape[0]

        # Expand slots to batch
        slots = rsm_slots.unsqueeze(0).expand(B, S, -1)    # (B, S, rsm_d)

        Q = self.q_proj(x)                                  # (B, T, d)
        K = self.k_proj(slots)                              # (B, S, d)
        V = self.v_proj(slots)                              # (B, S, d)

        # Reshape to multi-head
        def split_heads(t: torch.Tensor, seq: int) -> torch.Tensor:
            return t.view(B, seq, self.n_head, self.head_dim).transpose(1, 2)

        Q = split_heads(Q, T)                               # (B, nh, T, hd)
        K = split_heads(K, S)                               # (B, nh, S, hd)
        V = split_heads(V, S)

        att = (Q @ K.transpose(-2, -1)) * self.scale        # (B, nh, T, S)
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RSMLMBlock(nn.Module):
    """
    One transformer block:
      1. Causal self-attention  (current chunk only, no cache)
      2. RSM cross-attention    (read from fixed-size RSM slots)
      3. FFN
    """

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.norm1     = RMSNorm(cfg.d_model)
        self.self_attn = CausalSelfAttn(cfg.d_model, cfg.n_head, cfg.dropout)
        self.norm2     = RMSNorm(cfg.d_model)
        self.rsm_attn  = RSMCrossAttn(cfg.d_model, cfg.n_head, cfg.rsm_d_model, cfg.dropout)
        self.norm3     = RMSNorm(cfg.d_model)
        self.ffn       = FFN(cfg.d_model, cfg.ffn_dim, cfg.dropout)

    def forward(self, x: torch.Tensor, rsm_slots: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.rsm_attn(self.norm2(x), rsm_slots)
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RSMLM(nn.Module):
    """
    Language model that never needs a KV cache.

    The model:
      - Embeds the current chunk (up to block_size tokens)
      - Passes it through N RSMLMBlocks (local self-attn + RSM cross-attn + FFN)
      - Projects to vocabulary logits

    The RSM (passed as an argument to forward) provides all long-range context.
    Model parameters include the cross-attention projections, so the model
    LEARNS how to read and interpret RSM memory during training.

    Parameters
    ----------
    cfg : LMConfig
    """

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe  = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([RSMLMBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: embedding and lm_head share weights (GPT-2 style)
        self.lm_head.weight = self.wte.weight

        # Projection from LM hidden dim back to RSM dim for writing
        # Identity (no projection) when dims match
        if cfg.d_model != cfg.rsm_d_model:
            self.write_proj: Optional[nn.Linear] = nn.Linear(cfg.d_model, cfg.rsm_d_model, bias=False)
        else:
            self.write_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # RSM factory
    # ------------------------------------------------------------------

    def build_rsm(self, device: Optional[torch.device] = None) -> ResonantStateMemory:
        """Create a fresh RSM that matches this model's rsm_d_model / rsm_n_slots."""
        cfg = get_config({
            "d_model": self.cfg.rsm_d_model,
            "n_slots": self.cfg.rsm_n_slots,
            "write_alpha_high": 0.5,
            "write_alpha_mid":  0.3,
            "write_alpha_low":  0.05,
            "contradiction_sim_threshold": 0.85,
            "contradiction_threshold":     0.7,
            "eviction_importance_floor":   0.9,
            "eviction_confidence_floor":   0.95,
            "recency_decay_half_life":     10_000,
        })
        rsm = ResonantStateMemory(cfg)
        if device is not None:
            rsm = rsm.to(device)
        return rsm

    # ------------------------------------------------------------------
    # Write back to RSM after a forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def write_to_rsm(
        self,
        hidden: torch.Tensor,
        rsm: ResonantStateMemory,
        importance: float = 0.7,
    ) -> None:
        """
        Write the chunk summary into RSM after a forward pass.

        hidden     : (B, T, d_model) — last-layer hidden states (detached)
        rsm        : the ResonantStateMemory to update
        importance : write strength; use higher values for information-dense chunks
        """
        h = hidden.detach()
        h_mean = h.mean(dim=1)                                      # (B, d_model)
        if self.write_proj is not None:
            h_mean = self.write_proj(h_mean)                        # (B, rsm_d)
        for b in range(h_mean.shape[0]):
            v = l2_normalize(h_mean[b], dim=0)
            rsm.write(v, importance=importance)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        rsm: ResonantStateMemory,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        idx : (B, T) int64 — token indices, T <= block_size
        rsm : ResonantStateMemory — current memory state (summaries detached)

        Returns
        -------
        logits : (B, T, vocab_size)
        hidden : (B, T, d_model) — last-layer hidden states (for RSM write-back)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        # Token + positional embeddings
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))               # (B, T, d)

        # RSM slots are always detached — they are memory, not parameters
        rsm_slots = rsm.summaries.detach()                          # (n_slots, rsm_d)

        # Transformer blocks
        for block in self.layers:
            x = block(x, rsm_slots)

        hidden = x                                                   # (B, T, d)
        logits = self.lm_head(self.norm_f(x))                       # (B, T, vocab)
        return logits, hidden

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        rsm: ResonantStateMemory,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Generate max_new_tokens tokens autoregressively.

        Each step uses at most block_size context tokens (sliding window).
        RSM is updated after each generation step with the new hidden state.

        idx : (1, T) seed tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, hidden = self(idx_cond, rsm)
            logits = logits[:, -1, :] / temperature                 # (1, vocab)

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)       # (1, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # Update RSM with the new hidden state
            self.write_to_rsm(hidden, rsm)

        return idx

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        groups = {
            "embeddings":  ["wte", "wpe"],
            "self_attn":   ["self_attn"],
            "rsm_attn":    ["rsm_attn"],
            "ffn":         ["ffn"],
            "norms":       ["norm"],
            "lm_head":     ["lm_head"],
            "write_proj":  ["write_proj"],
        }
        counts = {}
        for label, prefixes in groups.items():
            total = 0
            for name, p in self.named_parameters():
                if p.requires_grad and any(pf in name for pf in prefixes):
                    total += p.numel()
            if total > 0:
                counts[label] = total
        return counts
