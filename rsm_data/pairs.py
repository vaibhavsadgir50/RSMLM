"""
PairMaker — generate training pairs from chunked documents.

Two pair types needed:

  1. Contrastive pairs  (Stage 1 — projection training)
     (anchor, positive, [negatives])
     anchor   : chunk i from doc D
     positive : chunk j from doc D, |i-j| <= max_span
     negatives: chunks from other documents

  2. Retrieval triples  (Stage 2 — query_proj training)
     (context_chunks, query_chunk, target_chunk)
     context : first split_frac of a document
     query   : a chunk from the second half
     target  : that same chunk's direct encoding is the retrieval target
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Contrastive pairs
# ---------------------------------------------------------------------------

def make_contrastive_pairs(
    docs: List[List[str]],
    max_span: int = 3,
    n_neg: int = 5,
    max_pairs_per_doc: int = 20,
    seed: int = 42,
) -> List[Tuple[str, str, List[str]]]:
    """
    Build (anchor, positive, negatives) triples.

    anchor   : chunk[i] from document D
    positive : chunk[j] from document D, 1 <= j-i <= max_span
    negatives: n_neg chunks drawn from other documents

    Returns List of (anchor_str, positive_str, [neg_str, ...])
    """
    rng = random.Random(seed)
    # Flatten all chunks with their doc index for negative sampling
    all_chunks: List[Tuple[int, str]] = [
        (doc_idx, chunk)
        for doc_idx, doc in enumerate(docs)
        for chunk in doc
    ]

    pairs: List[Tuple[str, str, List[str]]] = []

    for doc_idx, doc in enumerate(docs):
        if len(doc) < 2:
            continue
        doc_pairs = 0
        # Build positives from within-document proximity
        indices = list(range(len(doc)))
        rng.shuffle(indices)
        for i in indices:
            if doc_pairs >= max_pairs_per_doc:
                break
            # Positive: a chunk within max_span steps forward
            j_choices = [
                j for j in range(i + 1, min(i + max_span + 1, len(doc)))
            ]
            if not j_choices:
                continue
            j = rng.choice(j_choices)
            anchor   = doc[i]
            positive = doc[j]

            # Negatives: from other documents
            negs: List[str] = []
            attempts = 0
            while len(negs) < n_neg and attempts < n_neg * 10:
                attempts += 1
                d_idx, neg_chunk = rng.choice(all_chunks)
                if d_idx != doc_idx:
                    negs.append(neg_chunk)
            if len(negs) < n_neg:
                continue

            pairs.append((anchor, positive, negs))
            doc_pairs += 1

    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Retrieval triples
# ---------------------------------------------------------------------------

def make_retrieval_triples(
    docs: List[List[str]],
    split_frac: float = 0.6,
    n_queries_per_doc: int = 3,
    seed: int = 42,
) -> List[Tuple[List[str], str, str]]:
    """
    Build (context_chunks, query_chunk, target_chunk) triples.

    context_chunks : first split_frac of document (written to RSM)
    query_chunk    : a chunk from the second half (used to query RSM)
    target_chunk   : adjacent chunk to query (what RSM should help predict)

    The training signal: encode(query_chunk) → RSM.read() → should be
    close to encode(target_chunk).
    """
    rng = random.Random(seed)
    triples: List[Tuple[List[str], str, str]] = []

    for doc in docs:
        if len(doc) < 4:
            continue
        split = max(2, int(len(doc) * split_frac))
        context = doc[:split]
        query_pool = doc[split : -1]   # exclude last (needs a target after it)
        if not query_pool:
            continue

        # Sample n_queries_per_doc query positions
        indices = list(range(len(query_pool)))
        rng.shuffle(indices)
        for i in indices[:n_queries_per_doc]:
            query_idx_in_doc = split + i
            if query_idx_in_doc + 1 >= len(doc):
                continue
            query_chunk  = doc[query_idx_in_doc]
            target_chunk = doc[query_idx_in_doc + 1]   # next chunk is the target
            triples.append((context, query_chunk, target_chunk))

    rng.shuffle(triples)
    return triples


# ---------------------------------------------------------------------------
# Batch iterators
# ---------------------------------------------------------------------------

def pair_batches(
    pairs: List[Tuple[str, str, List[str]]],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[List[Tuple[str, str, List[str]]]]:
    """Yield batches of contrastive pairs."""
    idxs = list(range(len(pairs)))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    for start in range(0, len(idxs) - batch_size + 1, batch_size):
        yield [pairs[i] for i in idxs[start : start + batch_size]]


def triple_batches(
    triples: List[Tuple[List[str], str, str]],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[List[Tuple[List[str], str, str]]]:
    """Yield batches of retrieval triples."""
    idxs = list(range(len(triples)))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    for start in range(0, len(idxs) - batch_size + 1, batch_size):
        yield [triples[i] for i in idxs[start : start + batch_size]]
