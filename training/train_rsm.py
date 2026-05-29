"""
RSM standalone training — two stages.

Stage 1  Projection training (InfoNCE)
----------------------------------------
Trains the encoder projection (encoder_dim → d_model) so that chunks
from the same document map to nearby vectors in RSM embedding space,
while chunks from different documents map far apart.

Gradient path: loss → projection weights
RSM is not involved in Stage 1 — we're just learning a good embedding space
so that whatever gets written into RSM slots is semantically meaningful.

Stage 2  Retrieval training
----------------------------------------
Writes the first 60% of each document into RSM, then queries RSM
with second-half chunks. The retrieved vector should align with the
direct encoding of the target (next) chunk.

Gradient path: loss → query_proj (in RSM core) + projection weights
The RSM learns BOTH sides:
  - What to store: projection shapes what gets written via EMA
  - How to ask: query_proj shapes how queries are formulated

Run
---
  cd /Users/prem/UNI/RSM

  # Train on a local text file
  python -m training.train_rsm --source my_book.txt

  # Train on TinyShakespeare (auto-downloads)
  python -m training.train_rsm --source shakespeare

  # Train on HuggingFace wikitext-2
  python -m training.train_rsm --source hf:wikitext --hf_subset wikitext-2-raw-v1

  # Train on a directory of .txt files
  python -m training.train_rsm --source ./my_corpus/

  # Full options
  python -m training.train_rsm \\
      --source shakespeare \\
      --d_model 512 --n_slots 512 \\
      --s1_epochs 10 --s2_epochs 15 \\
      --batch_size 32 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from data.loader import TextLoader
from data.pairs import (
    make_contrastive_pairs,
    make_retrieval_triples,
    pair_batches,
    triple_batches,
)
from encoder.nomic_encoder import NomicTextEncoder
from encoder.projection import EncoderProjection
from rsm.core import ResonantStateMemory
from tensor_utils import l2_normalize
from training.checkpoint import load_checkpoint, save_checkpoint
from training.eval_rsm import eval_retrieval
from training.losses import info_nce, cosine_alignment


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

FALLBACK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Memory is the treasury and guardian of all things. "
    "To be, or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only way out is through. "
    "Knowledge is power, but wisdom is knowing when to use it. "
) * 500


def _load_source(
    source: str,
    chunk_size: int,
    overlap: int,
    hf_subset: Optional[str],
    hf_field: str,
    hf_split: str,
    max_docs: Optional[int],
) -> List[List[str]]:
    """Dispatch to the right TextLoader based on source string."""

    # HuggingFace dataset: "hf:dataset_name"
    if source.startswith("hf:"):
        dataset_name = source[3:]
        print(f"Loading HuggingFace dataset: {dataset_name} (subset={hf_subset}, split={hf_split})")
        return TextLoader.from_hf(
            dataset_name, split=hf_split, subset=hf_subset,
            text_field=hf_field, chunk_size=chunk_size,
            overlap=overlap, max_docs=max_docs,
        )

    # Built-in Shakespeare shortcut
    if source == "shakespeare":
        path = "data/shakespeare.txt"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(path):
            print(f"Downloading TinyShakespeare → {path}")
            try:
                urllib.request.urlretrieve(SHAKESPEARE_URL, path)
            except Exception as e:
                print(f"Download failed ({e}), using fallback text")
                with open(path, "w") as f:
                    f.write(FALLBACK_TEXT)
        return TextLoader.from_file(path, chunk_size, overlap)

    # Directory
    if os.path.isdir(source):
        print(f"Loading directory: {source}")
        return TextLoader.from_dir(source, chunk_size=chunk_size, overlap=overlap,
                                   max_files=max_docs)

    # Single file
    if os.path.isfile(source):
        print(f"Loading file: {source}")
        return TextLoader.from_file(source, chunk_size, overlap)

    raise ValueError(
        f"Unknown source: {source!r}. "
        "Use a file path, directory path, 'shakespeare', or 'hf:<dataset_name>'"
    )


# ---------------------------------------------------------------------------
# Stage 1 — projection training (InfoNCE)
# ---------------------------------------------------------------------------

def stage1_projection(
    encoder: nn.Module,
    projection: nn.Module,
    pairs: List[Tuple[str, str, List[str]]],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    temperature: float = 0.07,
    device: torch.device,
    val_pairs: Optional[List[Tuple[str, str, List[str]]]] = None,
    checkpoint_path: str = "checkpoints/rsm_s1_proj.pt",
) -> None:
    """
    Train projection with InfoNCE: related chunks (same document) closer,
    unrelated chunks (different document) farther.
    """
    encoder.eval()
    projection.train()
    opt = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    print(f"\n[Stage 1] Projection training | pairs={len(pairs)} | epochs={epochs} | lr={lr}")

    for epoch in range(epochs):
        total, steps = 0.0, 0
        for batch in pair_batches(pairs, batch_size, shuffle=True, seed=epoch):
            anchors   = [p[0] for p in batch]
            positives = [p[1] for p in batch]

            with torch.no_grad():
                ea = encoder(anchors).to(device)
                ep = encoder(positives).to(device)

            za = projection(ea)   # (B, d)
            zp = projection(ep)   # (B, d)

            loss = info_nce(za, zp, temperature=temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1

        sched.step()
        avg = total / max(steps, 1)

        val_str = ""
        if val_pairs:
            val_loss = _eval_contrastive(encoder, projection, val_pairs,
                                         batch_size, temperature, device)
            val_str = f" | val_loss {val_loss:.4f}"
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(checkpoint_path, projection.state_dict(),
                                opt.state_dict(), epoch, val_loss, {})
        else:
            if avg < best_loss:
                best_loss = avg
                save_checkpoint(checkpoint_path, projection.state_dict(),
                                opt.state_dict(), epoch, avg, {})

        print(f"  epoch {epoch+1:3d}/{epochs} | train_loss {avg:.4f}{val_str}")

    print(f"[Stage 1] Done. Best loss={best_loss:.4f} → {checkpoint_path}")


@torch.no_grad()
def _eval_contrastive(
    encoder: nn.Module,
    projection: nn.Module,
    pairs: List,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    projection.eval()
    total, steps = 0.0, 0
    for batch in pair_batches(pairs, batch_size, shuffle=False):
        ea = encoder([p[0] for p in batch]).to(device)
        ep = encoder([p[1] for p in batch]).to(device)
        za, zp = projection(ea), projection(ep)
        total += float(info_nce(za, zp, temperature=temperature).item())
        steps += 1
    projection.train()
    return total / max(steps, 1)


# ---------------------------------------------------------------------------
# Stage 2 — retrieval training (query_proj + projection)
# ---------------------------------------------------------------------------

def stage2_retrieval(
    encoder: nn.Module,
    projection: nn.Module,
    rsm_config: Dict[str, Any],
    triples: List[Tuple[List[str], str, str]],
    *,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: torch.device,
    val_triples: Optional[List] = None,
    checkpoint_path: str = "checkpoints/rsm_s2_retrieval.pt",
    s1_checkpoint: Optional[str] = None,
) -> None:
    """
    Train RSM's query_proj (and projection jointly) on retrieval task.

    For each triple (context_chunks, query_chunk, target_chunk):
      1. Write context_chunks into a fresh RSM via EMA (detached, no grad).
      2. Encode query_chunk → query_vec  (with grad through projection)
      3. retrieved = rsm.read(query_vec)  (grad flows through query_proj)
      4. Encode target_chunk → target_vec (with grad through projection)
      5. Loss = cosine_alignment(retrieved, target_vec)
                + info_nce(query_vec, target_vec)   ← keeps projection aligned

    Gradient paths:
      cosine_alignment: → retrieved → softmax(logits) → query_proj(query_vec)
                                                       → query_proj params ✓
                                                       → projection params ✓
      info_nce:         → query_vec, target_vec       → projection params ✓
    """
    encoder.eval()
    projection.train()

    # Build a single RSM instance with query_proj for training
    rsm_train = ResonantStateMemory(rsm_config).to(device)
    rsm_train.train()

    # Load stage 1 projection if available
    if s1_checkpoint and os.path.exists(s1_checkpoint):
        ck = load_checkpoint(s1_checkpoint)
        projection.load_state_dict(ck["model_state_dict"])
        print(f"[Stage 2] Loaded Stage 1 projection from {s1_checkpoint}")

    trainable = list(projection.parameters()) + list(rsm_train.query_proj.parameters())
    opt   = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    print(f"\n[Stage 2] Retrieval training | triples={len(triples)} | epochs={epochs} | lr={lr}")

    for epoch in range(epochs):
        total, steps = 0.0, 0

        for batch in triple_batches(triples, batch_size, shuffle=True, seed=epoch):
            batch_loss = torch.tensor(0.0, device=device)
            valid = 0

            for context_chunks, query_text, target_text in batch:
                # Fresh RSM state for this document (orthogonal init)
                rsm_train.summaries.fill_(0.0)
                torch.nn.init.orthogonal_(rsm_train.summaries)

                # --- Write context (no gradient through write path) ---
                with torch.no_grad():
                    ctx_vecs = projection(encoder(context_chunks).to(device))
                for v in ctx_vecs:
                    rsm_train.write(l2_normalize(v.detach(), dim=0), importance=0.7)

                # --- Encode query and target (gradient enabled) ---
                with torch.no_grad():
                    e_q = encoder([query_text]).to(device)
                    e_t = encoder([target_text]).to(device)
                q_vec = projection(e_q)[0]    # (d,)  — grad through projection
                t_vec = projection(e_t)[0]    # (d,)  — grad through projection

                # --- Retrieve from RSM (grad flows through query_proj) ---
                retrieved = rsm_train.read(q_vec)   # (d,) — grad via query_proj

                # --- Losses ---
                loss_retrieve = cosine_alignment(retrieved.unsqueeze(0),
                                                 t_vec.unsqueeze(0))
                loss_align = info_nce(q_vec.unsqueeze(0), t_vec.unsqueeze(0),
                                      temperature=0.07)
                batch_loss = batch_loss + loss_retrieve + 0.5 * loss_align
                valid += 1

            if valid == 0:
                continue

            batch_loss = batch_loss / valid
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += float(batch_loss.item())
            steps += 1

        sched.step()
        avg = total / max(steps, 1)

        val_str = ""
        if val_triples:
            val_loss = _eval_retrieval_loss(encoder, projection, rsm_train,
                                            rsm_config, val_triples, device)
            val_str = f" | val_loss {val_loss:.4f}"
            if val_loss < best_loss:
                best_loss = val_loss
                _save_s2(checkpoint_path, projection, rsm_train, opt, epoch, val_loss)
        else:
            if avg < best_loss:
                best_loss = avg
                _save_s2(checkpoint_path, projection, rsm_train, opt, epoch, avg)

        print(f"  epoch {epoch+1:3d}/{epochs} | train_loss {avg:.4f}{val_str}")

    print(f"[Stage 2] Done. Best loss={best_loss:.4f} → {checkpoint_path}")


@torch.no_grad()
def _eval_retrieval_loss(
    encoder, projection, rsm_train, rsm_config, triples, device
) -> float:
    projection.eval()
    rsm_train.eval()
    total, steps = 0.0, 0
    for context_chunks, query_text, target_text in triples[:200]:
        rsm_train.summaries.fill_(0.0)
        torch.nn.init.orthogonal_(rsm_train.summaries)
        ctx_vecs = projection(encoder(context_chunks).to(device))
        for v in ctx_vecs:
            rsm_train.write(l2_normalize(v, dim=0), importance=0.7)
        q_vec = projection(encoder([query_text]).to(device))[0]
        t_vec = projection(encoder([target_text]).to(device))[0]
        retrieved = rsm_train.read(q_vec)
        loss = cosine_alignment(retrieved.unsqueeze(0), t_vec.unsqueeze(0))
        total += float(loss.item())
        steps += 1
    projection.train()
    rsm_train.train()
    return total / max(steps, 1)


def _save_s2(path, projection, rsm_train, opt, epoch, val_loss):
    save_checkpoint(
        path,
        {"projection": projection.state_dict(),
         "query_proj": rsm_train.query_proj.state_dict()},
        opt.state_dict(), epoch, val_loss, {},
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_rsm(
    source: str,
    *,
    d_model: int = 512,
    n_slots: int = 512,
    encoder_dim: int = 768,
    chunk_size: int = 128,
    overlap: int = 32,
    s1_epochs: int = 10,
    s2_epochs: int = 15,
    batch_size: int = 32,
    lr_s1: float = 3e-4,
    lr_s2: float = 1e-4,
    val_frac: float = 0.1,
    checkpoint_dir: str = "checkpoints/",
    device: torch.device,
    use_transformers: bool = False,
    hf_subset: Optional[str] = None,
    hf_field: str = "text",
    hf_split: str = "train",
    max_docs: Optional[int] = None,
    skip_stage1: bool = False,
    skip_stage2: bool = False,
) -> Tuple[nn.Module, nn.Module, ResonantStateMemory]:
    """
    Full RSM training pipeline.

    Returns (encoder, projection, trained_rsm) ready for downstream LM training.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load data ---
    docs = _load_source(source, chunk_size, overlap, hf_subset, hf_field, hf_split, max_docs)
    if not docs:
        raise RuntimeError(f"No documents loaded from {source!r}")

    stats = TextLoader.stats(docs)
    print(f"\nDataset: {stats['n_docs']:,} docs | "
          f"{stats['n_chunks']:,} chunks | "
          f"{stats['total_words']:,.0f} words")

    # --- Train/val split ---
    val_n  = max(1, int(len(docs) * val_frac))
    val_docs   = docs[:val_n]
    train_docs = docs[val_n:]
    print(f"Train docs: {len(train_docs):,} | Val docs: {len(val_docs):,}")

    # --- Build encoder and projection ---
    rsm_config = get_config({
        "d_model": d_model,
        "n_slots": n_slots,
        "encoder_dim": encoder_dim,
    })
    encoder   = NomicTextEncoder(encoder_dim, use_transformers=use_transformers).to(device)
    projection = EncoderProjection(encoder_dim, d_model).to(device)
    encoder.eval()

    s1_ckpt = os.path.join(checkpoint_dir, "rsm_s1_proj.pt")
    s2_ckpt = os.path.join(checkpoint_dir, "rsm_s2_retrieval.pt")

    # --- Stage 1: projection ---
    if not skip_stage1:
        print("\nBuilding contrastive pairs for Stage 1...")
        train_pairs = make_contrastive_pairs(train_docs, max_span=4, n_neg=5,
                                             max_pairs_per_doc=20)
        val_pairs   = make_contrastive_pairs(val_docs, max_span=4, n_neg=5,
                                             max_pairs_per_doc=10)
        print(f"  train pairs: {len(train_pairs):,} | val pairs: {len(val_pairs):,}")
        stage1_projection(
            encoder, projection, train_pairs,
            epochs=s1_epochs, batch_size=batch_size, lr=lr_s1,
            device=device, val_pairs=val_pairs or None,
            checkpoint_path=s1_ckpt,
        )
    elif os.path.exists(s1_ckpt):
        ck = load_checkpoint(s1_ckpt)
        projection.load_state_dict(ck["model_state_dict"])
        print(f"[Stage 1] Skipped — loaded from {s1_ckpt}")

    # Eval retrieval after stage 1
    print("\nRetrieval eval after Stage 1:")
    scores = eval_retrieval(encoder, projection, rsm_config, val_docs,
                            n_docs=min(50, len(val_docs)), device=device)
    _print_scores(scores)

    # --- Stage 2: retrieval ---
    if not skip_stage2:
        print("\nBuilding retrieval triples for Stage 2...")
        train_triples = make_retrieval_triples(train_docs, n_queries_per_doc=3)
        val_triples   = make_retrieval_triples(val_docs,   n_queries_per_doc=3)
        print(f"  train triples: {len(train_triples):,} | val triples: {len(val_triples):,}")
        stage2_retrieval(
            encoder, projection, rsm_config, train_triples,
            epochs=s2_epochs, batch_size=batch_size // 2, lr=lr_s2,
            device=device, val_triples=val_triples or None,
            checkpoint_path=s2_ckpt, s1_checkpoint=s1_ckpt,
        )

    # Eval retrieval after stage 2
    print("\nRetrieval eval after Stage 2:")
    # Load trained query_proj back
    rsm_final = ResonantStateMemory(rsm_config).to(device)
    if os.path.exists(s2_ckpt):
        ck = load_checkpoint(s2_ckpt)
        projection.load_state_dict(ck["model_state_dict"]["projection"])
        rsm_final.query_proj.load_state_dict(ck["model_state_dict"]["query_proj"])

    scores = eval_retrieval(encoder, projection, rsm_config, val_docs,
                            n_docs=min(50, len(val_docs)), device=device)
    _print_scores(scores)

    print(f"\nRSM training complete.")
    print(f"  Stage 1 checkpoint : {s1_ckpt}")
    print(f"  Stage 2 checkpoint : {s2_ckpt}")
    print("  → Now use these to initialize LM training (--rsm_checkpoint)")

    return encoder, projection, rsm_final


def _print_scores(scores: Dict) -> None:
    for k, v in scores.items():
        if k == "n_queries":
            print(f"  {k:<14}: {int(v)}")
        else:
            print(f"  {k:<14}: {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train RSM on text data (2 stages)")

    # Data source
    p.add_argument("--source", default="shakespeare",
                   help="'shakespeare' | file path | dir path | 'hf:dataset_name'")
    p.add_argument("--hf_subset", default=None, help="HuggingFace dataset subset/config name")
    p.add_argument("--hf_field",  default="text", help="Field name in HF dataset")
    p.add_argument("--hf_split",  default="train")
    p.add_argument("--max_docs",  type=int, default=None)

    # Model dims
    p.add_argument("--d_model",     type=int,   default=512)
    p.add_argument("--n_slots",     type=int,   default=512)
    p.add_argument("--encoder_dim", type=int,   default=768)
    p.add_argument("--chunk_size",  type=int,   default=128)
    p.add_argument("--overlap",     type=int,   default=32)

    # Training
    p.add_argument("--s1_epochs",   type=int,   default=10)
    p.add_argument("--s2_epochs",   type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr_s1",       type=float, default=3e-4)
    p.add_argument("--lr_s2",       type=float, default=1e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--use_transformers", action="store_true",
                   help="Use real Nomic encoder (pip install transformers)")

    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--device", default="cpu")

    args = p.parse_args()
    device = torch.device(args.device)

    train_rsm(
        source=args.source,
        d_model=args.d_model,
        n_slots=args.n_slots,
        encoder_dim=args.encoder_dim,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        s1_epochs=args.s1_epochs,
        s2_epochs=args.s2_epochs,
        batch_size=args.batch_size,
        lr_s1=args.lr_s1,
        lr_s2=args.lr_s2,
        val_frac=args.val_frac,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        use_transformers=args.use_transformers,
        hf_subset=args.hf_subset,
        hf_field=args.hf_field,
        hf_split=args.hf_split,
        max_docs=args.max_docs,
        skip_stage1=args.skip_stage1,
        skip_stage2=args.skip_stage2,
    )


if __name__ == "__main__":
    main()
