"""
RSM retrieval evaluation.

Metrics
-------
  MRR@k    : Mean Reciprocal Rank — where does the true positive rank?
  Recall@k : Fraction of queries where the true positive is in top-k

Usage
-----
  from training.eval_rsm import eval_retrieval

  scores = eval_retrieval(encoder, projection, rsm_config, docs,
                          n_docs=100, top_k=[1, 5, 10])
  # → {"mrr@5": 0.42, "recall@1": 0.21, "recall@5": 0.55, ...}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rsm.core import ResonantStateMemory
from tensor_utils import l2_normalize


@torch.no_grad()
def eval_retrieval(
    encoder: nn.Module,
    projection: nn.Module,
    rsm_config: Dict,
    docs: List[List[str]],
    n_docs: int = 100,
    top_k: Tuple[int, ...] = (1, 5, 10),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure how well RSM retrieves related content after reading a document.

    Protocol per document:
      1. Write first 60% of chunks into RSM (context).
      2. Encode each second-half chunk (query).
      3. Encode the *next* chunk (target / positive).
      4. Query RSM with each query chunk.
      5. Rank: does the retrieved vector rank the true positive
              above all other second-half chunks?

    Returns dict with mrr@k and recall@k for each k in top_k.
    """
    device = device or torch.device("cpu")
    encoder.eval()
    projection.eval()

    def encode_texts(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            e = encoder(texts).to(device)
            return projection(e)   # (N, d_model)

    all_rr:     List[float] = []
    recall_hit: Dict[int, List[bool]] = {k: [] for k in top_k}

    eval_docs = docs[:n_docs]
    for doc in eval_docs:
        if len(doc) < 6:
            continue

        split   = max(3, int(len(doc) * 0.6))
        context = doc[:split]
        queries = doc[split:-1]         # chunks to query with
        targets = doc[split + 1:]       # next chunk = retrieval target

        if not queries:
            continue

        # Build RSM from context
        rsm = ResonantStateMemory(rsm_config).to(device)
        ctx_vecs = encode_texts(context)
        for v in ctx_vecs:
            rsm.write(l2_normalize(v, dim=0), importance=0.7)

        # Encode all second-half chunks (pool for ranking)
        query_vecs  = encode_texts(queries)   # (Q, d)
        target_vecs = encode_texts(targets)   # (Q, d) — one per query

        for qi in range(len(queries)):
            if qi >= len(targets):
                break

            # RSM retrieval for this query
            retrieved = rsm.read(query_vecs[qi])              # (d,) — has gradient path
            retrieved = l2_normalize(retrieved, dim=0).detach()

            # Compare retrieved to all target candidates
            pool = l2_normalize(target_vecs, dim=-1)          # (Q, d)
            sims = torch.mv(pool, retrieved)                  # (Q,)

            # True positive: target at position qi
            true_pos_sim = sims[qi].item()
            rank = int((sims > true_pos_sim).sum().item()) + 1  # 1-based rank

            rr = 1.0 / rank
            all_rr.append(rr)
            for k in top_k:
                recall_hit[k].append(rank <= k)

    if not all_rr:
        return {f"mrr@{max(top_k)}": 0.0, **{f"recall@{k}": 0.0 for k in top_k}}

    results: Dict[str, float] = {}
    max_k = max(top_k)
    results[f"mrr@{max_k}"] = sum(
        rr for rr, rank_within in zip(all_rr, recall_hit[max_k]) for _ in [None]
        if True
    ) / len(all_rr)
    # Simpler MRR (all queries)
    results[f"mrr@{max_k}"] = sum(all_rr) / len(all_rr)
    for k in top_k:
        results[f"recall@{k}"] = sum(recall_hit[k]) / len(recall_hit[k])
    results["n_queries"] = float(len(all_rr))
    return results
