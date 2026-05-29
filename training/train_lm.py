"""
Train RSMLM from scratch.

The model sees only a fixed chunk of tokens at a time (block_size).
RSM provides long-range context across chunks — no KV cache ever grows.

Training loop per document
--------------------------
  rsm.reset()                              # clear memory at doc boundary
  for chunk in sliding_window(doc):
      logits, hidden = model(chunk, rsm)   # forward: local attn + RSM read
      loss = cross_entropy(logits, target)
      loss.backward()
      optimizer.step()
      model.write_to_rsm(hidden, rsm)      # update memory with what we saw

Run
---
  cd /Users/prem/UNI/RSM
  python -m training.train_lm --preset tiny --steps 2000
  python -m training.train_lm --preset small --steps 5000 --device cuda
  python -m training.train_lm --preset base  --steps 10000 --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import get_config
from lm.model import RSMLM, LMConfig
from rsm.core import ResonantStateMemory
from training.checkpoint import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

FALLBACK_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd.
""" * 200   # repeat to give enough training signal


def load_text(path: str) -> str:
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    print(f"Downloading TinyShakespeare → {path}")
    try:
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        return open(path, encoding="utf-8").read()
    except Exception as e:
        print(f"Download failed ({e}), using fallback text")
        return FALLBACK_TEXT


def build_vocab(text: str) -> Tuple[List[str], dict, dict]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> List[int]:
    return [stoi[c] for c in text if c in stoi]


def make_chunks(
    tokens: List[int],
    block_size: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split token list into (x, y) pairs of length block_size."""
    chunks = []
    for i in range(0, len(tokens) - block_size, block_size):
        x = torch.tensor(tokens[i : i + block_size], dtype=torch.long, device=device)
        y = torch.tensor(tokens[i + 1 : i + block_size + 1], dtype=torch.long, device=device)
        chunks.append((x, y))
    return chunks


# ---------------------------------------------------------------------------
# Importance heuristic for RSM write-back
# ---------------------------------------------------------------------------

PUNCT = set(".!?;:")

def chunk_importance(tokens: torch.Tensor, itos: dict) -> float:
    """
    Simple importance score based on punctuation density.
    Sentences ending with . ! ? are more 'complete' → higher importance.
    Range [0.3, 0.9].
    """
    text = "".join(itos.get(int(t), "") for t in tokens)
    n_punct = sum(1 for c in text if c in PUNCT)
    ratio = n_punct / max(len(text), 1)
    return float(max(0.3, min(0.9, 0.3 + ratio * 10)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    cfg: LMConfig,
    text: str,
    stoi: dict,
    itos: dict,
    steps: int,
    lr: float,
    grad_clip: float,
    checkpoint_dir: str,
    device: torch.device,
    log_every: int = 100,
    save_every: int = 1000,
    eval_tokens: int = 200,
    warmup_steps: int = 100,
) -> None:
    tokens = encode(text, stoi)
    print(f"Dataset: {len(text):,} chars | {len(tokens):,} tokens | vocab={cfg.vocab_size}")

    # Split 90/10
    split = int(0.9 * len(tokens))
    train_tok = tokens[:split]
    val_tok   = tokens[split:]

    train_chunks = make_chunks(train_tok, cfg.block_size, device)
    val_chunks   = make_chunks(val_tok,   cfg.block_size, device)
    print(f"Train chunks: {len(train_chunks)} | Val chunks: {len(val_chunks)}")

    model = RSMLM(cfg).to(device)
    rsm   = model.build_rsm(device)

    print(f"\nModel: {model.count_parameters():,} parameters")
    for k, v in model.parameter_breakdown().items():
        print(f"  {k:<14}: {v:>10,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    # Cosine LR schedule with warmup
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "rsmlm_best.pt")

    best_val_loss = float("inf")
    chunk_idx = 0
    doc_boundary = 0             # reset RSM every N chunks (simulates doc boundary)
    doc_reset_every = max(1, len(train_chunks) // 10)

    t0 = time.time()
    model.train()

    for step in range(steps):
        # Update LR
        new_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # RSM reset at document boundaries
        if chunk_idx % doc_reset_every == 0:
            rsm.summaries.fill_(0.0)
            torch.nn.init.orthogonal_(rsm.summaries)

        x, y = train_chunks[chunk_idx % len(train_chunks)]
        chunk_idx += 1

        logits, hidden = model(x.unsqueeze(0), rsm)           # (1, T, vocab), (1, T, d)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            y.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Update RSM with chunk summary (no gradient)
        imp = chunk_importance(x, itos)
        model.write_to_rsm(hidden, rsm, importance=imp)

        if (step + 1) % log_every == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"step {step + 1:5d}/{steps} | loss {loss.item():.4f} | "
                f"lr {new_lr:.2e} | {elapsed:.1f}s"
            )
            t0 = time.time()

        if (step + 1) % save_every == 0:
            val_loss = evaluate(model, rsm, val_chunks, cfg, device, n_chunks=50)
            print(f"  val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_path,
                    {"model": model.state_dict(), "rsm": rsm.summaries},
                    optimizer.state_dict(),
                    step,
                    val_loss,
                    {"lm_config": cfg.__dict__, "vocab_size": cfg.vocab_size},
                )
                print(f"  saved checkpoint (val_loss={val_loss:.4f})")
            model.train()

    # Final sample
    print("\n--- generated sample ---")
    sample = generate_sample(model, rsm, itos, stoi, device, cfg, n=eval_tokens)
    print(sample)
    print("------------------------")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: RSMLM,
    rsm: ResonantStateMemory,
    chunks: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: LMConfig,
    device: torch.device,
    n_chunks: int = 50,
) -> float:
    model.eval()
    # Fresh RSM for eval (don't contaminate train state)
    eval_rsm = model.build_rsm(device)
    total, count = 0.0, 0
    for x, y in chunks[:n_chunks]:
        logits, hidden = model(x.unsqueeze(0), eval_rsm)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        total += float(loss.item())
        count += 1
        model.write_to_rsm(hidden, eval_rsm, importance=0.7)
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(
    model: RSMLM,
    rsm: ResonantStateMemory,
    itos: dict,
    stoi: dict,
    device: torch.device,
    cfg: LMConfig,
    prompt: str = "\n",
    n: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    model.eval()
    seed_ids = [stoi.get(c, 0) for c in prompt]
    idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
    gen_rsm = model.build_rsm(device)
    out_idx = model.generate(idx, gen_rsm, max_new_tokens=n, temperature=temperature, top_k=top_k)
    return "".join(itos[int(t)] for t in out_idx[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train RSMLM — LLM with RSM memory, no KV cache")
    p.add_argument("--preset",     choices=["tiny", "small", "base"], default="small",
                   help="Model size preset")
    p.add_argument("--steps",      type=int,   default=5000)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--data",       type=str,   default="data/shakespeare.txt",
                   help="Path to text file (downloads TinyShakespeare if missing)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument("--log_every",  type=int,   default=100)
    p.add_argument("--save_every", type=int,   default=500)
    p.add_argument("--device",     type=str,   default="cpu")
    # Override individual dims
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--n_layer",    type=int,   default=None)
    p.add_argument("--n_head",     type=int,   default=None)
    p.add_argument("--block_size", type=int,   default=None)
    p.add_argument("--rsm_slots",  type=int,   default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
    text = load_text(args.data)
    chars, stoi, itos = build_vocab(text)

    # Build config from preset then apply overrides
    preset_fn = {"tiny": LMConfig.tiny, "small": LMConfig.small, "base": LMConfig.base}
    cfg = preset_fn[args.preset](vocab_size=len(chars))
    if args.d_model:    cfg.d_model    = args.d_model
    if args.n_layer:    cfg.n_layer    = args.n_layer
    if args.n_head:     cfg.n_head     = args.n_head
    if args.block_size: cfg.block_size = args.block_size
    if args.rsm_slots:
        cfg.rsm_n_slots = args.rsm_slots
        cfg.rsm_d_model = cfg.d_model   # keep dims aligned

    print(f"Config: {cfg}")

    train(
        cfg=cfg,
        text=text,
        stoi=stoi,
        itos=itos,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
