"""Tests for RSMLM — architecture, RSM integration, generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lm.model import RSMLM, LMConfig
from rsm.core import ResonantStateMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(**kwargs) -> LMConfig:
    defaults = dict(vocab_size=32, d_model=64, n_layer=2, n_head=4,
                    ffn_dim=128, block_size=16, dropout=0.0,
                    rsm_d_model=64, rsm_n_slots=32)
    defaults.update(kwargs)
    return LMConfig(**defaults)


def _model_and_rsm(**kwargs):
    cfg = _tiny_cfg(**kwargs)
    model = RSMLM(cfg)
    rsm = model.build_rsm()
    return model, rsm, cfg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def test_build_tiny_preset():
    cfg = LMConfig.tiny(vocab_size=65)
    model = RSMLM(cfg)
    assert model.count_parameters() > 0

def test_build_small_preset():
    cfg = LMConfig.small(vocab_size=65)
    model = RSMLM(cfg)
    n = model.count_parameters()
    assert n > 1_000_000

def test_parameter_breakdown_keys():
    model, _, _ = _model_and_rsm()
    breakdown = model.parameter_breakdown()
    assert "self_attn" in breakdown
    assert "rsm_attn"  in breakdown
    assert "ffn"       in breakdown

def test_build_rsm_shape():
    model, rsm, cfg = _model_and_rsm()
    assert rsm.summaries.shape == (cfg.rsm_n_slots, cfg.rsm_d_model)

def test_weight_tying():
    model, _, _ = _model_and_rsm()
    assert model.lm_head.weight is model.wte.weight


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_output_shapes():
    model, rsm, cfg = _model_and_rsm()
    B, T = 2, cfg.block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, hidden = model(idx, rsm)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert hidden.shape == (B, T, cfg.d_model)

def test_forward_single_token():
    model, rsm, cfg = _model_and_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, 1))
    logits, hidden = model(idx, rsm)
    assert logits.shape == (1, 1, cfg.vocab_size)

def test_forward_exceeds_block_size_raises():
    import pytest
    model, rsm, cfg = _model_and_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx, rsm)

def test_logits_are_finite():
    model, rsm, cfg = _model_and_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits, _ = model(idx, rsm)
    assert torch.isfinite(logits).all()

def test_different_rsm_states_give_different_logits():
    """Verify RSM actually influences output."""
    model, rsm1, cfg = _model_and_rsm()
    rsm2 = model.build_rsm()

    # Write random vectors into rsm1 to make it different from fresh rsm2
    for _ in range(20):
        v = torch.randn(cfg.rsm_d_model)
        from tensor_utils import l2_normalize
        rsm1.write(l2_normalize(v, dim=0), importance=0.8)

    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits1, _ = model(idx, rsm1)
        logits2, _ = model(idx, rsm2)

    assert not torch.allclose(logits1, logits2), \
        "RSM state must influence logits"


# ---------------------------------------------------------------------------
# RSM cross-attention: no gradient through RSM buffers
# ---------------------------------------------------------------------------

def test_rsm_slots_receive_no_gradient():
    """RSM summaries are buffers — they must not accumulate gradients."""
    model, rsm, cfg = _model_and_rsm()
    rsm.summaries.requires_grad_(False)     # buffers shouldn't need this, but be explicit

    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits, _ = model(idx, rsm)
    loss = logits.sum()
    loss.backward()

    assert rsm.summaries.grad is None, "RSM summaries must not receive gradients"

def test_model_params_receive_gradients():
    model, rsm, cfg = _model_and_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits, _ = model(idx, rsm)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size),
                           torch.randint(0, cfg.vocab_size, (cfg.block_size,)))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0, "Model parameters must receive gradients"


# ---------------------------------------------------------------------------
# RSM write-back
# ---------------------------------------------------------------------------

def test_write_to_rsm_changes_state():
    model, rsm, cfg = _model_and_rsm()
    before = rsm.summaries.clone()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, hidden = model(idx, rsm)
    model.write_to_rsm(hidden, rsm, importance=0.8)
    assert not torch.allclose(rsm.summaries, before), \
        "RSM summaries should change after write_to_rsm"

def test_write_to_rsm_batch():
    model, rsm, cfg = _model_and_rsm()
    B = 3
    idx = torch.randint(0, cfg.vocab_size, (B, cfg.block_size))
    with torch.no_grad():
        _, hidden = model(idx, rsm)
    model.write_to_rsm(hidden, rsm, importance=0.7)


# ---------------------------------------------------------------------------
# RSM reset between documents
# ---------------------------------------------------------------------------

def test_rsm_reset_clears_state():
    model, rsm, cfg = _model_and_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, hidden = model(idx, rsm)
    model.write_to_rsm(hidden, rsm, importance=0.9)
    after_write = rsm.summaries.clone()

    # Simulate document boundary: reset RSM
    rsm.summaries.fill_(0.0)
    torch.nn.init.orthogonal_(rsm.summaries)

    assert not torch.allclose(rsm.summaries, after_write)


# ---------------------------------------------------------------------------
# Projection when d_model != rsm_d_model
# ---------------------------------------------------------------------------

def test_mismatched_dims_uses_write_proj():
    cfg = _tiny_cfg(d_model=64, rsm_d_model=32)
    model = RSMLM(cfg)
    assert model.write_proj is not None
    rsm = model.build_rsm()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        logits, hidden = model(idx, rsm)
    model.write_to_rsm(hidden, rsm, importance=0.7)
    assert logits.shape == (1, cfg.block_size, cfg.vocab_size)

def test_matched_dims_no_write_proj():
    cfg = _tiny_cfg(d_model=64, rsm_d_model=64)
    model = RSMLM(cfg)
    assert model.write_proj is None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generate_length():
    model, rsm, cfg = _tiny_cfg(), None, None
    cfg = _tiny_cfg()
    model = RSMLM(cfg)
    rsm = model.build_rsm()
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(seed, rsm, max_new_tokens=10, temperature=1.0, top_k=0)
    assert out.shape == (1, 14)  # 4 seed + 10 new

def test_generate_stays_in_vocab():
    cfg = _tiny_cfg()
    model = RSMLM(cfg)
    rsm = model.build_rsm()
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(seed, rsm, max_new_tokens=20)
    assert out.max().item() < cfg.vocab_size
    assert out.min().item() >= 0

def test_generate_updates_rsm():
    cfg = _tiny_cfg()
    model = RSMLM(cfg)
    rsm = model.build_rsm()
    before = rsm.summaries.clone()
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    model.generate(seed, rsm, max_new_tokens=5)
    assert not torch.allclose(rsm.summaries, before), \
        "generate() should update RSM after each token"


# ---------------------------------------------------------------------------
# Loss decreases with gradient steps (sanity check)
# ---------------------------------------------------------------------------

def test_loss_decreases_over_steps():
    cfg = _tiny_cfg()
    model = RSMLM(cfg)
    rsm = model.build_rsm()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (cfg.block_size,))

    losses = []
    for _ in range(10):
        logits, hidden = model(x, rsm)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.write_to_rsm(hidden, rsm, importance=0.7)
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
