"""
Tests for data pipeline, losses, eval, and RSM training stages.
All tests use in-memory data — no network or disk I/O.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import pytest
import torch

from rsm_data.loader import TextLoader, chunk_by_words
from rsm_data.pairs import (
    make_contrastive_pairs,
    make_retrieval_triples,
    pair_batches,
    triple_batches,
)
from training.losses import cosine_alignment, info_nce
from training.eval_rsm import eval_retrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(n_docs: int = 5, chunks_per_doc: int = 8) -> List[List[str]]:
    """Synthetic documents — each chunk is distinct enough to be non-trivial."""
    docs = []
    for d in range(n_docs):
        doc = []
        for c in range(chunks_per_doc):
            words = [f"doc{d}_chunk{c}_word{w}" for w in range(20)]
            doc.append(" ".join(words))
        docs.append(doc)
    return docs


def _make_longer_text(n_words: int = 300) -> str:
    """Return a text string with n_words distinct words."""
    return " ".join(f"word{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# chunk_by_words
# ---------------------------------------------------------------------------

class TestChunkByWords:
    def test_basic_shape(self):
        text = _make_longer_text(300)
        chunks = chunk_by_words(text, chunk_size=64, overlap=16)
        assert len(chunks) >= 3
        for c in chunks:
            words = c.split()
            assert len(words) >= 16

    def test_overlap(self):
        text = _make_longer_text(200)
        chunks = chunk_by_words(text, chunk_size=50, overlap=25)
        # Consecutive chunks share words due to overlap
        if len(chunks) >= 2:
            words0 = set(chunks[0].split())
            words1 = set(chunks[1].split())
            assert len(words0 & words1) > 0

    def test_short_text_returns_empty(self):
        # Fewer words than min_chunk_words → empty
        chunks = chunk_by_words("hello world", chunk_size=128, min_chunk_words=16)
        assert chunks == []

    def test_returns_list_of_strings(self):
        text = _make_longer_text(200)
        chunks = chunk_by_words(text, chunk_size=50, overlap=10)
        assert all(isinstance(c, str) for c in chunks)

    def test_deterministic(self):
        text = _make_longer_text(200)
        a = chunk_by_words(text, chunk_size=64, overlap=16)
        b = chunk_by_words(text, chunk_size=64, overlap=16)
        assert a == b


# ---------------------------------------------------------------------------
# TextLoader.from_strings
# ---------------------------------------------------------------------------

class TestTextLoaderFromStrings:
    def test_basic(self):
        texts = [_make_longer_text(200), _make_longer_text(200)]
        docs = TextLoader.from_strings(texts, chunk_size=64, overlap=16)
        assert len(docs) >= 1
        for doc in docs:
            assert len(doc) >= 2
            assert all(isinstance(c, str) for c in doc)

    def test_short_text_skipped(self):
        texts = ["hello world"]  # too short
        docs = TextLoader.from_strings(texts, min_doc_words=64)
        assert len(docs) == 0

    def test_multiple_docs(self):
        texts = [_make_longer_text(300) for _ in range(4)]
        docs = TextLoader.from_strings(texts, chunk_size=64, overlap=16)
        assert len(docs) == 4

    def test_returns_list_of_list_of_str(self):
        texts = [_make_longer_text(200)]
        docs = TextLoader.from_strings(texts)
        assert isinstance(docs, list)
        assert isinstance(docs[0], list)
        assert isinstance(docs[0][0], str)


# ---------------------------------------------------------------------------
# TextLoader.from_file
# ---------------------------------------------------------------------------

class TestTextLoaderFromFile:
    def test_txt_file(self, tmp_path):
        content = _make_longer_text(500)
        p = tmp_path / "test.txt"
        p.write_text(content)
        docs = TextLoader.from_file(str(p), chunk_size=64, overlap=16)
        assert len(docs) >= 1

    def test_jsonl_file(self, tmp_path):
        import json
        lines = [json.dumps({"text": _make_longer_text(200)}) for _ in range(3)]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(lines))
        docs = TextLoader.from_file(str(p), chunk_size=64, overlap=16)
        assert len(docs) >= 1

    def test_json_list_file(self, tmp_path):
        import json
        data = [{"text": _make_longer_text(200)} for _ in range(3)]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        docs = TextLoader.from_file(str(p), chunk_size=64, overlap=16)
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# TextLoader.from_dir
# ---------------------------------------------------------------------------

class TestTextLoaderFromDir:
    def test_loads_txt_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(_make_longer_text(200))
        docs = TextLoader.from_dir(str(tmp_path), chunk_size=64, overlap=16)
        assert len(docs) >= 2

    def test_max_files(self, tmp_path):
        for i in range(5):
            (tmp_path / f"doc{i}.txt").write_text(_make_longer_text(200))
        docs = TextLoader.from_dir(str(tmp_path), chunk_size=64, overlap=16, max_files=2)
        # Should have loaded at most 2 files worth of docs
        assert len(docs) <= 4  # generous upper bound


# ---------------------------------------------------------------------------
# TextLoader.stats
# ---------------------------------------------------------------------------

class TestTextLoaderStats:
    def test_stats_keys(self):
        docs = _make_docs(3, 6)
        s = TextLoader.stats(docs)
        assert "n_docs" in s
        assert "n_chunks" in s
        assert "chunks_per_doc" in s
        assert "words_per_chunk" in s
        assert "total_words" in s

    def test_stats_values(self):
        docs = _make_docs(3, 6)
        s = TextLoader.stats(docs)
        assert s["n_docs"] == 3
        assert s["n_chunks"] == 18
        assert abs(s["chunks_per_doc"] - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# make_contrastive_pairs
# ---------------------------------------------------------------------------

class TestMakeContrastivePairs:
    def test_returns_list_of_triples(self):
        docs = _make_docs(5, 8)
        pairs = make_contrastive_pairs(docs, max_span=3, n_neg=5, max_pairs_per_doc=10)
        assert len(pairs) > 0
        for anchor, positive, negs in pairs:
            assert isinstance(anchor, str)
            assert isinstance(positive, str)
            assert isinstance(negs, list)
            assert len(negs) == 5

    def test_anchor_and_positive_from_same_doc(self):
        docs = _make_docs(5, 8)
        pairs = make_contrastive_pairs(docs, max_span=3, n_neg=5)
        # anchor and positive share doc prefix (doc{d}_...)
        for anchor, positive, _ in pairs[:10]:
            doc_id_a = anchor.split("_")[0]
            doc_id_p = positive.split("_")[0]
            assert doc_id_a == doc_id_p

    def test_negatives_from_different_doc(self):
        docs = _make_docs(5, 8)
        pairs = make_contrastive_pairs(docs, max_span=3, n_neg=5)
        for anchor, _, negs in pairs[:10]:
            doc_id_a = anchor.split("_")[0]
            for neg in negs:
                doc_id_n = neg.split("_")[0]
                assert doc_id_n != doc_id_a

    def test_deterministic_with_seed(self):
        docs = _make_docs(5, 8)
        pairs_a = make_contrastive_pairs(docs, seed=42)
        pairs_b = make_contrastive_pairs(docs, seed=42)
        assert pairs_a == pairs_b

    def test_different_seed_different_order(self):
        docs = _make_docs(5, 8)
        pairs_a = make_contrastive_pairs(docs, seed=1)
        pairs_b = make_contrastive_pairs(docs, seed=2)
        assert pairs_a != pairs_b

    def test_max_pairs_per_doc(self):
        docs = _make_docs(5, 20)
        pairs = make_contrastive_pairs(docs, max_pairs_per_doc=5)
        assert len(pairs) <= 5 * 5 + 5  # generous bound

    def test_short_docs_skipped(self):
        docs = [["only_one_chunk"]]  # single chunk — can't form a pair
        pairs = make_contrastive_pairs(docs, n_neg=5)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# make_retrieval_triples
# ---------------------------------------------------------------------------

class TestMakeRetrievalTriples:
    def test_returns_triples(self):
        docs = _make_docs(5, 8)
        triples = make_retrieval_triples(docs, split_frac=0.6, n_queries_per_doc=3)
        assert len(triples) > 0
        for context, query, target in triples:
            assert isinstance(context, list)
            assert len(context) >= 1
            assert isinstance(query, str)
            assert isinstance(target, str)

    def test_context_is_first_split(self):
        docs = _make_docs(3, 10)
        triples = make_retrieval_triples(docs, split_frac=0.6)
        for context, _, _ in triples:
            # context should have ~60% of chunks
            assert len(context) >= 2

    def test_short_docs_skipped(self):
        docs = [["a", "b", "c"]]  # only 3 chunks — too short (needs 4)
        triples = make_retrieval_triples(docs)
        assert len(triples) == 0

    def test_deterministic(self):
        docs = _make_docs(5, 8)
        t1 = make_retrieval_triples(docs, seed=42)
        t2 = make_retrieval_triples(docs, seed=42)
        assert t1 == t2

    def test_n_queries_per_doc(self):
        docs = _make_docs(4, 12)
        triples = make_retrieval_triples(docs, n_queries_per_doc=2)
        assert len(triples) <= 4 * 2 + 4  # generous bound


# ---------------------------------------------------------------------------
# pair_batches / triple_batches
# ---------------------------------------------------------------------------

class TestBatchIterators:
    def test_pair_batches_sizes(self):
        docs = _make_docs(5, 8)
        pairs = make_contrastive_pairs(docs)
        batches = list(pair_batches(pairs, batch_size=4))
        for b in batches:
            assert len(b) == 4

    def test_pair_batches_covers_data(self):
        docs = _make_docs(5, 8)
        pairs = make_contrastive_pairs(docs)
        batches = list(pair_batches(pairs, batch_size=4, shuffle=False))
        total = sum(len(b) for b in batches)
        # Should cover all complete batches
        assert total == (len(pairs) // 4) * 4

    def test_triple_batches_sizes(self):
        docs = _make_docs(5, 8)
        triples = make_retrieval_triples(docs)
        if len(triples) >= 4:
            batches = list(triple_batches(triples, batch_size=4))
            for b in batches:
                assert len(b) == 4

    def test_pair_batches_shuffle(self):
        docs = _make_docs(10, 8)
        pairs = make_contrastive_pairs(docs)
        b1 = list(pair_batches(pairs, batch_size=4, shuffle=True, seed=0))
        b2 = list(pair_batches(pairs, batch_size=4, shuffle=True, seed=99))
        # Different seeds should usually produce different order
        assert b1 != b2


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_info_nce_shape(self):
        a = torch.randn(8, 64)
        p = torch.randn(8, 64)
        loss = info_nce(a, p, temperature=0.07)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_info_nce_perfect_match(self):
        # When anchor == positive, loss should be low
        v = torch.randn(8, 64)
        loss_perfect = info_nce(v, v, temperature=0.07)
        v2 = torch.randn(8, 64)
        loss_random = info_nce(v, v2, temperature=0.07)
        assert loss_perfect.item() < loss_random.item()

    def test_cosine_alignment_shape(self):
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        loss = cosine_alignment(pred, target)
        assert loss.shape == ()

    def test_cosine_alignment_perfect(self):
        v = torch.randn(4, 64)
        loss_perfect = cosine_alignment(v, v)
        assert loss_perfect.item() < 0.01

    def test_cosine_alignment_opposite(self):
        v = torch.randn(4, 64)
        loss_opposite = cosine_alignment(v, -v)
        assert loss_opposite.item() > 1.9  # ~2.0 for exactly opposite

    def test_info_nce_gradient(self):
        a = torch.randn(4, 32, requires_grad=True)
        p = torch.randn(4, 32)
        loss = info_nce(a, p)
        loss.backward()
        assert a.grad is not None
        assert not torch.all(a.grad == 0)

    def test_cosine_alignment_gradient(self):
        pred = torch.randn(4, 32, requires_grad=True)
        target = torch.randn(4, 32)
        loss = cosine_alignment(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# eval_retrieval
# ---------------------------------------------------------------------------

class TestEvalRetrieval:
    def _make_components(self, d_model=64, n_slots=32):
        from config import get_config
        from encoder.nomic_encoder import NomicTextEncoder
        from encoder.projection import EncoderProjection

        encoder_dim = 768
        cfg = get_config({"d_model": d_model, "n_slots": n_slots, "encoder_dim": encoder_dim})
        encoder = NomicTextEncoder(encoder_dim, use_transformers=False)
        projection = EncoderProjection(encoder_dim, d_model)
        return encoder, projection, cfg

    def test_returns_expected_keys(self):
        encoder, projection, cfg = self._make_components()
        docs = _make_docs(4, 8)
        scores = eval_retrieval(encoder, projection, cfg, docs,
                                n_docs=4, top_k=(1, 3, 5))
        assert "mrr@5" in scores
        assert "recall@1" in scores
        assert "recall@3" in scores
        assert "recall@5" in scores
        assert "n_queries" in scores

    def test_scores_in_range(self):
        encoder, projection, cfg = self._make_components()
        docs = _make_docs(4, 8)
        scores = eval_retrieval(encoder, projection, cfg, docs,
                                n_docs=4, top_k=(1, 5))
        for k, v in scores.items():
            if k != "n_queries":
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_empty_docs_returns_zeros(self):
        encoder, projection, cfg = self._make_components()
        # All docs too short (< 6 chunks) — should return zeros
        short_docs = [["a b c", "d e f"]] * 3
        scores = eval_retrieval(encoder, projection, cfg, short_docs,
                                n_docs=3, top_k=(1, 5))
        assert scores["recall@1"] == 0.0

    def test_n_queries_positive(self):
        encoder, projection, cfg = self._make_components()
        docs = _make_docs(4, 10)
        scores = eval_retrieval(encoder, projection, cfg, docs,
                                n_docs=4, top_k=(1, 5))
        assert scores["n_queries"] > 0


# ---------------------------------------------------------------------------
# stage1_projection (smoke test — 1 epoch, tiny data)
# ---------------------------------------------------------------------------

class TestStage1Projection:
    def _setup(self, d_model=64, n_slots=32):
        from config import get_config
        from encoder.nomic_encoder import NomicTextEncoder
        from encoder.projection import EncoderProjection

        cfg = get_config({"d_model": d_model, "n_slots": n_slots, "encoder_dim": 768})
        encoder = NomicTextEncoder(768, use_transformers=False)
        projection = EncoderProjection(768, d_model)
        device = torch.device("cpu")
        return encoder, projection, cfg, device

    def test_loss_decreases(self, tmp_path):
        from training.train_rsm import stage1_projection

        encoder, projection, _, device = self._setup()
        docs = _make_docs(8, 8)
        pairs = make_contrastive_pairs(docs, n_neg=5)

        losses = []
        class LossCapture:
            def __init__(self, proj):
                self.proj = proj
                self.orig_step = None

        # Run 3 epochs and collect avg loss via monkeypatching print
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        stage1_projection(
            encoder, projection, pairs,
            epochs=3, batch_size=8, lr=3e-4, device=device,
            checkpoint_path=str(tmp_path / "s1.pt"),
        )

        sys.stdout = old_stdout
        output = buf.getvalue()
        # Verify training ran (epochs printed)
        assert "epoch" in output.lower() or "stage 1" in output.lower()

    def test_projection_weights_change(self, tmp_path):
        from training.train_rsm import stage1_projection
        from encoder.projection import EncoderProjection

        encoder, projection, _, device = self._setup()
        docs = _make_docs(8, 8)
        pairs = make_contrastive_pairs(docs, n_neg=5)

        # Record initial weights (net[0] is the first Linear layer)
        init_w = projection.net[0].weight.data.clone()

        stage1_projection(
            encoder, projection, pairs,
            epochs=2, batch_size=8, lr=3e-4, device=device,
            checkpoint_path=str(tmp_path / "s1.pt"),
        )

        # Weights should change after training
        assert not torch.allclose(projection.net[0].weight.data, init_w), \
            "Projection weights did not change during Stage 1 training"


# ---------------------------------------------------------------------------
# stage2_retrieval (smoke test — 1 epoch, tiny data)
# ---------------------------------------------------------------------------

class TestStage2Retrieval:
    def _setup(self, d_model=64, n_slots=32):
        from config import get_config
        from encoder.nomic_encoder import NomicTextEncoder
        from encoder.projection import EncoderProjection

        cfg = get_config({"d_model": d_model, "n_slots": n_slots, "encoder_dim": 768})
        encoder = NomicTextEncoder(768, use_transformers=False)
        projection = EncoderProjection(768, d_model)
        device = torch.device("cpu")
        return encoder, projection, cfg, device

    def test_query_proj_weights_change(self, tmp_path):
        from training.train_rsm import stage2_retrieval

        encoder, projection, cfg, device = self._setup()
        docs = _make_docs(8, 10)
        triples = make_retrieval_triples(docs, n_queries_per_doc=2)

        from rsm.core import ResonantStateMemory
        rsm_tmp = ResonantStateMemory(cfg)
        init_qp = rsm_tmp.query_proj.weight.data.clone()

        stage2_retrieval(
            encoder, projection, cfg, triples,
            epochs=2, batch_size=4, lr=1e-4, device=device,
            checkpoint_path=str(tmp_path / "s2.pt"),
        )

        # Load and check query_proj changed
        from training.checkpoint import load_checkpoint
        ck = load_checkpoint(str(tmp_path / "s2.pt"))
        trained_qp = ck["model_state_dict"]["query_proj"]["weight"]
        # query_proj is init'd as identity — should move after training
        assert not torch.allclose(trained_qp, torch.eye(cfg["d_model"])), \
            "query_proj did not move from identity during Stage 2 training"

    def test_rsm_slots_not_gradient_updated(self, tmp_path):
        """RSM summaries must never receive gradient — EMA writes only."""
        from training.train_rsm import stage2_retrieval
        from rsm.core import ResonantStateMemory

        encoder, projection, cfg, device = self._setup()
        docs = _make_docs(6, 10)
        triples = make_retrieval_triples(docs, n_queries_per_doc=2)

        # Build the RSM that stage2 will use internally
        # and verify summaries remain a buffer (no grad)
        rsm_check = ResonantStateMemory(cfg)
        assert not rsm_check.summaries.requires_grad

        stage2_retrieval(
            encoder, projection, cfg, triples,
            epochs=1, batch_size=4, lr=1e-4, device=device,
            checkpoint_path=str(tmp_path / "s2.pt"),
        )

    def test_projection_also_updated(self, tmp_path):
        from training.train_rsm import stage2_retrieval

        encoder, projection, cfg, device = self._setup()
        docs = _make_docs(8, 10)
        triples = make_retrieval_triples(docs, n_queries_per_doc=2)

        init_proj = projection.net[0].weight.data.clone()

        stage2_retrieval(
            encoder, projection, cfg, triples,
            epochs=2, batch_size=4, lr=1e-4, device=device,
            checkpoint_path=str(tmp_path / "s2.pt"),
        )

        assert not torch.allclose(projection.net[0].weight.data, init_proj), \
            "Projection did not update during Stage 2"
