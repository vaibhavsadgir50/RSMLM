"""Checkpoint I/O (model, optimizer, epoch, val loss, config snapshot)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    epoch: int,
    val_loss: float,
    config: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
