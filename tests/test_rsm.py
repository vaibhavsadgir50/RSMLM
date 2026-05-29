"""Tests for the 3-tier RSM (Layer1 buffer, Layer2 core, Layer3 longterm, RSMemory)."""

from __future__ import annotations

import time

import pytest
import torch

from config import get_config
from rsm.buffer import ImportanceBuffer
from rsm.core import ResonantStateMemory
from rsm.importance_scorer import ImportanceScorer
from rsm.longterm import LongTermStore
from rsm.memory import RSMemory
from tensor_utils import l2_normalize


# ---------------------------------------------------------------------------
# ImportanceScorer
# ---------------------------------------------------------------------------

def test_scorer_explicit():
    assert ImportanceScorer.score("You must never do that") >= 0.9

def test_scorer_ack():
    assert ImportanceScorer.score("ok thanks") <= 0.1

def test_scorer_default():
    assert ImportanceScorer.score("random chat") == 0.5

def test_scorer_numerical():
    assert ImportanceScorer.score("Revenue was $4.2 million") >= 0.85

def test_scorer_question():
    s = ImportanceScorer.score("what time is it?")
    assert s <= 0.4

def test_scorer_empty():
    assert ImportanceScorer.score("") == 0.5


# ---------------------------------------------------------------------------
# Layer 1 — ImportanceBuffer
# ---------------------------------------------------------------------------

def test_buffer_write_and_len():
    buf = ImportanceBuffer(capacity=5, d_model=8)
    for i in range(5):
        v = l2_normalize(torch.randn(8), dim=0)
        buf.write(v, f"text {i}", 0.5)
    assert len(buf) == 5

def test_buffer_eviction_keeps_highest_importance():
    buf = ImportanceBuffer(capacity=3, d_model=8)
    for i, imp in enumerate([0.2, 0.9, 0.4, 0.8]):
        v = torch.zeros(8)
        v[i % 8] = 1.0
        v = l2_normalize(v, dim=0)
        buf.write(v, f"t{i}", imp)
    assert len(buf) == 3
    # The 0.2 entry should have been evicted
    importances = [e.importance for e in buf._entries]
    assert min(importances) >= 0.4

def test_buffer_read_empty():
    buf = ImportanceBuffer(capacity=5, d_model=8)
    q = l2_normalize(torch.randn(8), dim=0)
    assert buf.read(q) == []

def test_buffer_read_returns_relevant():
    buf = ImportanceBuffer(capacity=10, d_model=4)
    target = l2_normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    opposite = l2_normalize(torch.tensor([-1.0, 0.0, 0.0, 0.0]), dim=0)
    buf.write(target.clone(), "relevant", 0.9)
    buf.write(opposite.clone(), "irrelevant", 0.9)
    results = buf.read(target, top_k=1)
    assert len(results) == 1
    assert results[0].text == "relevant"

def test_buffer_wrong_shape_raises():
    buf = ImportanceBuffer(capacity=5, d_model=8)
    with pytest.raises(ValueError):
        buf.write(torch.randn(4), "bad", 0.5)


# ---------------------------------------------------------------------------
# Layer 2 — ResonantStateMemory core
# ---------------------------------------------------------------------------

def test_rsm_read_returns_unit_vector():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    q = l2_normalize(torch.randn(32), dim=0)
    ctx = rsm.read(q)
    assert ctx.shape == (32,)
    assert abs(float(ctx.norm()) - 1.0) < 1e-4

def test_rsm_write_returns_metadata():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    x = l2_normalize(torch.randn(32), dim=0)
    meta = rsm.write(x, importance=0.6)
    assert "primary_slot" in meta
    assert "contradiction" in meta
    assert "erg_resolution_required" in meta

def test_rsm_version_increments_on_write():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    x = l2_normalize(torch.randn(32), dim=0)
    meta = rsm.write(x, importance=0.7)
    slot = meta["primary_slot"]
    assert int(rsm.version[slot].item()) >= 1

def test_rsm_pointer_set_and_get():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    rsm.set_pointer(0, "some-key")
    assert rsm.pointer(0) == "some-key"

def test_rsm_wrong_shape_raises():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    with pytest.raises(ValueError):
        rsm.read(torch.randn(16))
    with pytest.raises(ValueError):
        rsm.write(torch.randn(16), importance=0.5)

def test_rsm_high_importance_updates_more():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    x = l2_normalize(torch.randn(32), dim=0)
    before = rsm.summaries.clone()
    rsm.write(x, importance=0.95)
    delta_high = (rsm.summaries - before).norm()
    rsm2 = ResonantStateMemory(cfg)
    rsm2.summaries.data.copy_(before)
    rsm2.write(x, importance=0.01)
    delta_low = (rsm2.summaries - before).norm()
    assert delta_high > delta_low

def test_rsm_evict_returns_slot_or_none():
    cfg = get_config({"d_model": 32, "n_slots": 16})
    rsm = ResonantStateMemory(cfg)
    result = rsm.evict()
    # fresh RSM with zero importance — may or may not evict depending on floors
    assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# Layer 3 — LongTermStore
# ---------------------------------------------------------------------------

def test_longterm_high_importance_stored():
    lt = LongTermStore(d_model=16, topic_sim_threshold=0.5)
    v = l2_normalize(torch.randn(16), dim=0)
    key = lt.extract_sync(v, "important fact", 0.9)
    assert key != ""
    assert lt.num_topics() == 1

def test_longterm_low_importance_not_stored():
    lt = LongTermStore(d_model=16, topic_sim_threshold=0.5)
    v = l2_normalize(torch.randn(16), dim=0)
    key = lt.extract_sync(v, "low importance", 0.5)
    assert key == ""
    assert lt.num_topics() == 0

def test_longterm_versioning():
    lt = LongTermStore(d_model=16, topic_sim_threshold=0.5)
    v1 = l2_normalize(torch.randn(16), dim=0)
    k = lt.extract_sync(v1, "fact a", 0.9)
    v2 = l2_normalize(v1 + 0.01 * torch.randn_like(v1), dim=0)
    lt.extract_sync(v2, "fact a revised", 0.9)
    chain = lt.get_chain(k)
    assert len(chain) == 2
    assert chain[-1].superseded is False
    assert chain[0].superseded is True

def test_longterm_query_returns_relevant():
    lt = LongTermStore(d_model=16, topic_sim_threshold=0.3)
    v_a = l2_normalize(torch.tensor([1.0] + [0.0] * 15), dim=0)
    v_b = l2_normalize(torch.tensor([0.0, 1.0] + [0.0] * 14), dim=0)
    lt.extract_sync(v_a, "topic A", 0.9)
    lt.extract_sync(v_b, "topic B", 0.9)
    results = lt.query(v_a, top_k=1)
    assert len(results) == 1
    assert results[0].text == "topic A"

def test_longterm_async_extract():
    lt = LongTermStore(d_model=16, topic_sim_threshold=0.5)
    v = l2_normalize(torch.randn(16), dim=0)
    completed = []
    lt.schedule_extract(v, "async fact", 0.95, on_complete=lambda k: completed.append(k))
    time.sleep(0.1)
    assert len(completed) == 1
    assert completed[0] != ""


# ---------------------------------------------------------------------------
# RSMemory (full 3-tier)
# ---------------------------------------------------------------------------

def test_rsmemory_forward_shape():
    cfg = get_config({"d_model": 64, "n_slots": 32, "buffer_capacity": 10})
    def enc(t: str) -> torch.Tensor:
        x = torch.randn(64)
        x[0] = float(len(t))
        return l2_normalize(x, dim=0)
    mem = RSMemory(cfg, encode_fn=enc)
    out = mem.forward("query text", "input must follow rule X")
    assert out.shape == (64,)
    assert abs(float(out.norm()) - 1.0) < 1e-4

def test_rsmemory_buffer_receives_writes():
    cfg = get_config({"d_model": 32, "n_slots": 16, "buffer_capacity": 5})
    def enc(t: str) -> torch.Tensor:
        return l2_normalize(torch.randn(32), dim=0)
    mem = RSMemory(cfg, encode_fn=enc)
    for i in range(3):
        mem.forward("q", f"input text {i}")
    assert len(mem.buffer) == 3

def test_rsmemory_high_importance_hits_longterm():
    cfg = get_config({"d_model": 32, "n_slots": 16, "buffer_capacity": 5, "importance_threshold": 0.85})
    def enc(t: str) -> torch.Tensor:
        return l2_normalize(torch.randn(32), dim=0)
    mem = RSMemory(cfg, encode_fn=enc)
    mem.forward("q", "you must always save the model")
    time.sleep(0.15)  # let async thread complete
    assert mem.longterm.num_topics() >= 1

def test_rsmemory_erg_feedback_write():
    cfg = get_config({"d_model": 32, "n_slots": 16, "buffer_capacity": 5})
    def enc(t: str) -> torch.Tensor:
        return l2_normalize(torch.randn(32), dim=0)
    mem = RSMemory(cfg, encode_fn=enc)
    conclusion = l2_normalize(torch.randn(32), dim=0)
    meta = mem.erg_feedback_write(conclusion, confidence=0.9)
    assert "primary_slot" in meta

def test_rsmemory_fallback_encoder_works():
    cfg = get_config({"d_model": 256, "n_slots": 64, "buffer_capacity": 10, "encoder_dim": 768})
    mem = RSMemory(cfg, use_transformers=False)
    out = mem.forward("what is the rule?", "you must never skip validation")
    assert out.shape == (256,)
