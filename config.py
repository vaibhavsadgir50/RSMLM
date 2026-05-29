"""Central hyperparameters for the 3-tier RSM."""

from __future__ import annotations

from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    # Dimensions
    "d_model": 1024,
    "encoder_dim": 768,
    "n_slots": 512,
    # Layer 1 — importance buffer
    "buffer_capacity": 50,
    "importance_threshold": 0.85,
    # Layer 2 — RSM slots
    "write_alpha_high": 0.5,
    "write_alpha_mid": 0.3,
    "write_alpha_low": 0.05,
    "contradiction_sim_threshold": 0.85,
    "contradiction_threshold": 0.7,
    "eviction_importance_floor": 0.9,
    "eviction_confidence_floor": 0.95,
    "recency_decay_half_life": 10_000,
    # Stage 1 training (projection)
    "s1_lr": 3e-4,
    "s1_epochs": 10,
    "s1_batch_size": 32,
    # Stage 2 training (full RSMemory)
    "s2_lr": 3e-5,
    "s2_epochs": 30,
    "s2_batch_size": 32,
    "s2_val_target": 0.80,
    # Checkpoints
    "checkpoint_dir": "checkpoints/",
    "s1_checkpoint": "rsm_stage1_projection.pt",
    "s2_checkpoint": "rsm_stage2_best.pt",
}


def get_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg
