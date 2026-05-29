"""
RSM training entry point — two stages.

Stage 1: Train the encoder projection (768 → d_model) with InfoNCE.
Stage 2: Train full RSMemory forward pass with contrastive alignment.

Run from /Users/prem/UNI/RSM/:
    python -m training.train --stage 1
    python -m training.train --stage 2
    python -m training.train --stage all
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import get_config
from encoder.nomic_encoder import NomicTextEncoder
from encoder.projection import EncoderProjection
from rsm.memory import RSMemory
from tensor_utils import l2_normalize
from training.checkpoint import load_checkpoint, save_checkpoint
from training.losses import cosine_alignment, info_nce


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _default_pairs(n: int) -> List[Tuple[str, str]]:
    """Generate synthetic (anchor, positive) text pairs for projection training."""
    pairs: List[Tuple[str, str]] = []
    for i in range(n):
        pairs.append((f"concept {i} alpha", f"concept {i} beta"))
        pairs.append((f"rule must hold {i}", f"constraint never violate {i}"))
        pairs.append((f"The value is {i * 3.14:.2f} dollars", f"Price equals {i * 3.14:.2f} USD"))
    return pairs


def _default_context_pairs(n: int) -> List[Tuple[str, str, str]]:
    """Generate (query, input, expected-similar-input) triples for stage 2."""
    triples: List[Tuple[str, str, str]] = []
    topics = [
        ("machine learning", "Neural networks learn from data", "Deep learning uses backpropagation"),
        ("memory", "RSM stores facts in slots", "The memory system has 512 fixed slots"),
        ("rules", "You must always save checkpoints", "Never skip the validation step"),
        ("numbers", "Revenue was $4.2 million last quarter", "Sales reached $4.2M in Q3"),
    ]
    for i in range(n):
        q, inp, pos = topics[i % len(topics)]
        triples.append((q, inp, pos))
    return triples


# ---------------------------------------------------------------------------
# Stage 1 — projection training
# ---------------------------------------------------------------------------

def run_stage1(
    config: Dict[str, Any],
    text_pairs: Optional[List[Tuple[str, str]]] = None,
    device: Optional[torch.device] = None,
    use_transformers: bool = False,
) -> str:
    device = device or torch.device("cpu")
    text_pairs = text_pairs or _default_pairs(100)

    encoder = NomicTextEncoder(
        encoder_dim=int(config["encoder_dim"]),
        use_transformers=use_transformers,
    ).to(device)
    encoder.eval()

    projection = EncoderProjection(
        encoder_dim=int(config["encoder_dim"]),
        d_model=int(config["d_model"]),
    ).to(device)

    opt = torch.optim.AdamW(projection.parameters(), lr=float(config["s1_lr"]))
    batch_size = int(config["s1_batch_size"])
    epochs = int(config["s1_epochs"])

    ckpt_dir = config["checkpoint_dir"]
    out_path = os.path.join(ckpt_dir, config["s1_checkpoint"])

    print(f"\n[Stage 1] Training projection {config['encoder_dim']}→{config['d_model']}")
    print(f"  pairs={len(text_pairs)}, epochs={epochs}, batch={batch_size}, lr={config['s1_lr']}")

    best_loss = float("inf")
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for start in range(0, len(text_pairs), batch_size):
            batch = text_pairs[start : start + batch_size]
            if len(batch) < 2:
                continue
            left = [a for a, _ in batch]
            right = [b for _, b in batch]
            with torch.no_grad():
                el = encoder(left).to(device)
                er = encoder(right).to(device)
            zl = projection(el)
            zr = projection(er)
            loss = info_nce(zl, zr, temperature=0.07)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        val_loss = total / max(steps, 1)
        print(f"  epoch {epoch + 1:3d}/{epochs} | loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(out_path, projection.state_dict(), opt.state_dict(), epoch, val_loss, config)

    print(f"[Stage 1] Done. Best loss={best_loss:.4f}. Checkpoint: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 2 — full RSMemory training
# ---------------------------------------------------------------------------

def run_stage2(
    config: Dict[str, Any],
    triples: Optional[List[Tuple[str, str, str]]] = None,
    device: Optional[torch.device] = None,
    s1_checkpoint: Optional[str] = None,
    use_transformers: bool = False,
) -> str:
    device = device or torch.device("cpu")
    triples = triples or _default_context_pairs(200)

    mem = RSMemory(config, use_transformers=use_transformers).to(device)

    # Load stage 1 projection weights if available
    s1_path = s1_checkpoint or os.path.join(config["checkpoint_dir"], config["s1_checkpoint"])
    if os.path.exists(s1_path) and mem.text_projection is not None:
        ck = load_checkpoint(s1_path)
        mem.text_projection.load_state_dict(ck["model_state_dict"])
        print(f"[Stage 2] Loaded Stage 1 projection from {s1_path}")

    # Freeze encoder; train projection + (optionally) nothing else in stage 2
    if mem.text_encoder is not None:
        for p in mem.text_encoder.parameters():
            p.requires_grad = False
        mem.text_encoder.eval()

    trainable = [p for p in mem.parameters() if p.requires_grad]
    if not trainable:
        print("[Stage 2] Warning: no trainable parameters — check projection requires_grad")

    opt = torch.optim.AdamW(trainable, lr=float(config["s2_lr"]))
    batch_size = int(config["s2_batch_size"])
    epochs = int(config["s2_epochs"])

    ckpt_dir = config["checkpoint_dir"]
    out_path = os.path.join(ckpt_dir, config["s2_checkpoint"])

    print(f"\n[Stage 2] Full RSMemory training")
    print(f"  triples={len(triples)}, epochs={epochs}, batch={batch_size}, lr={config['s2_lr']}")

    best_loss = float("inf")
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for start in range(0, len(triples), batch_size):
            batch = triples[start : start + batch_size]
            if len(batch) < 2:
                continue
            # Encode all texts
            queries = [t[0] for t in batch]
            inputs = [t[1] for t in batch]
            positives = [t[2] for t in batch]

            if mem.text_encoder is None:
                continue
            with torch.no_grad():
                eq = mem.text_encoder(queries).to(device)
                ei = mem.text_encoder(inputs).to(device)
                ep = mem.text_encoder(positives).to(device)
            assert mem.text_projection is not None
            zq = mem.text_projection(eq)
            zi = mem.text_projection(ei)
            zp = mem.text_projection(ep)

            # Contrastive: projected input should align with projected positive
            loss = info_nce(zi, zp, temperature=0.07)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1

        val_loss = total / max(steps, 1)
        print(f"  epoch {epoch + 1:3d}/{epochs} | loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            if mem.text_projection is not None:
                save_checkpoint(
                    out_path,
                    mem.text_projection.state_dict(),
                    opt.state_dict(),
                    epoch,
                    val_loss,
                    config,
                )

    print(f"[Stage 2] Done. Best loss={best_loss:.4f}. Checkpoint: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train the 3-tier RSM")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--encoder_dim", type=int, default=768)
    p.add_argument("--n_slots", type=int, default=512)
    p.add_argument("--s1_epochs", type=int, default=10)
    p.add_argument("--s2_epochs", type=int, default=30)
    p.add_argument("--s1_lr", type=float, default=3e-4)
    p.add_argument("--s2_lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument("--use_transformers", action="store_true",
                   help="Load real Nomic encoder (requires pip install transformers)")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    config = get_config({
        "d_model": args.d_model,
        "encoder_dim": args.encoder_dim,
        "n_slots": args.n_slots,
        "s1_epochs": args.s1_epochs,
        "s2_epochs": args.s2_epochs,
        "s1_lr": args.s1_lr,
        "s2_lr": args.s2_lr,
        "s1_batch_size": args.batch_size,
        "s2_batch_size": args.batch_size,
        "checkpoint_dir": args.checkpoint_dir,
    })

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.stage in ("1", "all"):
        run_stage1(config, device=device, use_transformers=args.use_transformers)

    if args.stage in ("2", "all"):
        run_stage2(config, device=device, use_transformers=args.use_transformers)


if __name__ == "__main__":
    main()
